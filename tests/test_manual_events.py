"""ADR-19 wiring tests for src/pipeline/manual_events.py (CLAUDE.md §14).

`bucket_annotations_by_take` and `build_manual_identity_by_take` are exercised directly (the
latter with `associate_annotation_to_track` monkeypatched -- it does a real video decode).
`run_manual_events_pipeline_for_video` gets one "acceptance #2" SMOKE test with every I/O-heavy
boundary (profiling, detect/track, video decode, the annotated-video render, and per-player
output writing) monkeypatched out, per the plan's own scoping: this proves the pipeline's own
CONTROL FLOW is wired correctly (sidecar parsed -> bucketed by take -> grouped by jersey number ->
written per player, with the right identity status and goal_reason), not that ffmpeg/decode
genuinely runs -- the real end-to-end run is deferred to the owner's own annotated video.
"""

from __future__ import annotations

from src.common.io import load_yaml
from src.common.types import Annotation, BBox, DetectionClass, EventType, Take, Track, TrackBox
from src.pipeline import manual_events as me_mod


def _take(tid: int, t_start: float, t_end: float) -> Take:
    return Take(id=tid, t_start=t_start, t_end=t_end, frame_start=0, frame_end=100, kind="main")


def _annotation(t: float, jersey: int = 2, colour: str = "white") -> Annotation:
    return Annotation(
        t=t,
        jersey_number=jersey,
        team_colour=colour,
        action_phrase="takes a touch",
        event_type=EventType.TOUCH,
        raw_line=f"Min {t} player #{jersey} in {colour} takes a touch",
    )


# ---------------------------------------------------------------------------
# bucket_annotations_by_take (pure)
# ---------------------------------------------------------------------------


def test_bucket_annotations_by_take_assigns_correct_take():
    takes = [_take(0, 0.0, 10.0), _take(1, 10.0, 20.0)]
    anns = [_annotation(5.0), _annotation(15.0)]
    by_take, n_unassigned = me_mod.bucket_annotations_by_take(anns, takes)
    assert n_unassigned == 0
    assert by_take[0] == [anns[0]]
    assert by_take[1] == [anns[1]]


def test_bucket_annotations_by_take_counts_out_of_range_as_unassigned():
    takes = [_take(0, 0.0, 10.0)]
    anns = [_annotation(500.0)]
    by_take, n_unassigned = me_mod.bucket_annotations_by_take(anns, takes)
    assert by_take == {}
    assert n_unassigned == 1


# ---------------------------------------------------------------------------
# build_manual_identity_by_take (associate_annotation_to_track monkeypatched -- real decode)
# ---------------------------------------------------------------------------


def _track(tid: int, t: float) -> Track:
    box = TrackBox(frame_index=0, t=t, bbox=BBox(x1=0, y1=0, x2=20, y2=100), conf=0.9)
    return Track(id=tid, take_id=0, boxes=[box], dominant_class=DetectionClass.PLAYER)


def _box_at(t: float, cx: float, cy: float, height: float = 100.0) -> TrackBox:
    half_h = height / 2.0
    return TrackBox(
        frame_index=int(round(t * 10)),
        t=t,
        bbox=BBox(x1=cx - 20.0, y1=cy - half_h, x2=cx + 20.0, y2=cy + half_h),
        conf=0.9,
    )


def _selection_cfg() -> dict:
    return load_yaml("configs/highlights.yaml")["selection"]


def test_build_manual_identity_by_take_verified_when_association_succeeds(monkeypatch):
    take = _take(0, 0.0, 10.0)
    track = _track(1, 1.0)
    anns = [_annotation(1.0)]

    monkeypatch.setattr(
        me_mod,
        "associate_annotation_to_track",
        lambda *a, **k: (
            1,
            5.0,
            {"arrow_track_id": None, "arrow_distance_px": None, "agreement": "colour_only"},
        ),
    )

    identity_by_take, debug = me_mod.build_manual_identity_by_take(
        [take],
        {0: [track]},
        {0: anns},
        "fake.mp4",
        {"annotations": {"track_association": {}}, "hardware": {"decode": {}}},
    )
    assert identity_by_take[0].status == "verified"
    assert identity_by_take[0].jersey_number == 2
    assert identity_by_take[0].location_track_ids == [1]
    assert identity_by_take[0].location_method == "manual_annotation_colour_match"
    assert "0:1.00" in debug
    assert debug["0:1.00"]["agreement"] == "colour_only"


def test_build_manual_identity_by_take_no_identity_when_association_fails(monkeypatch):
    take = _take(0, 0.0, 10.0)
    track = _track(1, 1.0)
    anns = [_annotation(1.0)]

    monkeypatch.setattr(
        me_mod,
        "associate_annotation_to_track",
        lambda *a, **k: (
            None,
            55.0,
            {"arrow_track_id": None, "arrow_distance_px": None, "agreement": "no_confident_signal"},
        ),
    )

    identity_by_take, _debug = me_mod.build_manual_identity_by_take(
        [take],
        {0: [track]},
        {0: anns},
        "fake.mp4",
        {"annotations": {"track_association": {}}, "hardware": {"decode": {}}},
    )
    assert identity_by_take == {}


def test_build_manual_identity_by_take_arrow_fallback_recorded_in_debug(monkeypatch):
    """When colour found nothing but the arrow signal did, `build_manual_identity_by_take` must
    still end up with a verified identity (via the arrow's own fallback pick) AND the debug trail
    must say so (`agreement == "arrow_fallback"`) -- Golden Rule 5 traceability, not just a bare
    track id with no explanation of which signal actually produced it."""
    take = _take(0, 0.0, 10.0)
    track = _track(1, 1.0)
    anns = [_annotation(1.0)]

    monkeypatch.setattr(
        me_mod,
        "associate_annotation_to_track",
        lambda *a, **k: (
            1,
            42.0,
            {"arrow_track_id": 1, "arrow_distance_px": 42.0, "agreement": "arrow_fallback"},
        ),
    )

    identity_by_take, debug = me_mod.build_manual_identity_by_take(
        [take],
        {0: [track]},
        {0: anns},
        "fake.mp4",
        {"annotations": {"track_association": {}}, "hardware": {"decode": {}}},
        arrow_hints=["not empty -- just needs to be truthy for this monkeypatched test"],
    )
    assert identity_by_take[0].status == "verified"
    assert debug["0:1.00"]["agreement"] == "arrow_fallback"
    assert debug["0:1.00"]["arrow_track_id"] == 1


# ---------------------------------------------------------------------------
# Bug fix 2026-08-31: identity LOCK across a take -- once a jersey number resolves confidently to
# a track's stitched identity chain (src.track.continuity.build_take_identities), every LATER
# annotation of the SAME jersey number in the SAME take must be restricted to that chain, never
# re-opened to an independent whole-field search. Confirmed on real data
# (output/chelsea_burnley_target10/annotation_report.json): jersey #10 resolved to raw track 208
# at t=57.0s and a DIFFERENT real track 210 at t=60.0s -- a genuine cross-player identity swap.
# ---------------------------------------------------------------------------


def test_build_manual_identity_by_take_locks_onto_same_track_despite_closer_distractor(
    monkeypatch,
):
    """Two candidate tracks exist near the SECOND annotation: track 100 (the same physical player
    as the first annotation) and track 200 (a distractor, spatially far from 100 so it never
    stitches into 100's identity chain). An UNRESTRICTED per-instant search (the pre-fix
    behaviour) would pick 200 at the second instant -- proven directly below. The FIX must instead
    lock onto 100's identity chain at the first annotation and restrict the second annotation's
    search to that chain only, so it resolves to 100 again, never 200."""
    take = _take(0, 0.0, 10.0)
    # track_a: one continuous physical player, visible near BOTH annotation instants.
    track_a = Track(id=100, take_id=0, boxes=[_box_at(1.0, 50, 50), _box_at(5.0, 52, 50)])
    # track_b: a distractor only near the second instant, far enough away (450 units, height 100)
    # that build_take_identities' own stitch_max_dist (3.0 bbox-heights) rejects joining it to A.
    track_b = Track(id=200, take_id=0, boxes=[_box_at(5.0, 500, 500)])
    anns = [_annotation(1.0, jersey=10), _annotation(5.0, jersey=10)]

    def fake_associate(video_path, take, candidates, ann, *args, **kwargs):
        ids = {tr.id for tr in candidates}
        if ids == {100, 200}:
            # UNRESTRICTED search (what the pre-fix code always did): simulate an independent
            # colour match landing on the DIFFERENT track at the later instant.
            if ann.t < 3.0:
                return (
                    100,
                    5.0,
                    {"arrow_track_id": None, "arrow_distance_px": None, "agreement": "colour_only"},
                )
            return (
                200,
                2.0,
                {"arrow_track_id": None, "arrow_distance_px": None, "agreement": "colour_only"},
            )
        if ids == {100}:
            return (
                100,
                5.0,
                {"arrow_track_id": None, "arrow_distance_px": None, "agreement": "colour_only"},
            )
        raise AssertionError(f"unexpected candidate id set {ids} for ann.t={ann.t}")

    monkeypatch.setattr(me_mod, "associate_annotation_to_track", fake_associate)

    # Sanity-check the premise: an unrestricted search at the later instant really does pick the
    # distractor (200) -- exactly what the old, memory-less per-annotation code did.
    assert fake_associate("fake.mp4", take, [track_a, track_b], anns[1])[0] == 200

    configs = {
        "annotations": {"track_association": {}},
        "hardware": {"decode": {}},
        "highlights": {"selection": _selection_cfg()},
    }
    identity_by_take, debug = me_mod.build_manual_identity_by_take(
        [take], {0: [track_a, track_b]}, {0: anns}, "fake.mp4", configs
    )

    # NEW code: jersey #10 locks onto track 100's identity chain at t=1.0; the t=5.0 annotation is
    # RESTRICTED to that same chain and must resolve to 100 again, never the distractor.
    assert debug["0:1.00"]["track_id"] == 100
    assert debug["0:5.00"]["track_id"] == 100
    assert debug["0:5.00"]["locked_chain_id"] == 100
    assert identity_by_take[0].location_track_ids == [100]


def test_build_manual_identity_by_take_honest_caption_only_when_locked_identity_absent(
    monkeypatch,
):
    """Once jersey #10 locks onto track 100's chain at the first annotation, if NO member of that
    chain has a confident candidate near a LATER annotation's own instant, the honest result is
    caption-only (`track_id=None`) -- never a fallback to an unrestricted whole-field search (that
    would silently reopen the exact cross-player swap the lock exists to prevent)."""
    take = _take(0, 0.0, 10.0)
    # track_a only appears near the FIRST annotation; track_b only near the second, far enough
    # away in time (gap 4.0s > stitch_max_gap_s 1.5s) that they never stitch into one chain.
    track_a = Track(id=100, take_id=0, boxes=[_box_at(1.0, 50, 50)])
    track_b = Track(id=200, take_id=0, boxes=[_box_at(5.0, 500, 500)])
    anns = [_annotation(1.0, jersey=10), _annotation(5.0, jersey=10)]

    def fake_associate(video_path, take, candidates, ann, *args, **kwargs):
        ids = {tr.id for tr in candidates}
        if ids == {100, 200} and ann.t < 3.0:
            return (
                100,
                5.0,
                {"arrow_track_id": None, "arrow_distance_px": None, "agreement": "colour_only"},
            )
        if ids == {100} and ann.t >= 3.0:
            # locked chain (100) has no confident candidate near this later instant.
            return (
                None,
                None,
                {
                    "arrow_track_id": None,
                    "arrow_distance_px": None,
                    "agreement": "no_confident_signal",
                },
            )
        raise AssertionError(f"unexpected candidate id set {ids} for ann.t={ann.t}")

    monkeypatch.setattr(me_mod, "associate_annotation_to_track", fake_associate)

    configs = {
        "annotations": {"track_association": {}},
        "hardware": {"decode": {}},
        "highlights": {"selection": _selection_cfg()},
    }
    identity_by_take, debug = me_mod.build_manual_identity_by_take(
        [take], {0: [track_a, track_b]}, {0: anns}, "fake.mp4", configs
    )

    assert debug["0:1.00"]["track_id"] == 100
    assert debug["0:5.00"]["track_id"] is None
    assert debug["0:5.00"]["agreement"] == "locked_identity_absent_at_instant"
    assert debug["0:5.00"]["locked_chain_id"] == 100
    # Track 200 (the distractor) never enters the union of confidently-associated ids.
    assert identity_by_take[0].location_track_ids == [100]


def test_build_manual_identity_by_take_two_distinct_jerseys_lock_independently(monkeypatch):
    """Requirement (CLAUDE.md task): a take with MULTIPLE distinct jersey numbers must lock each
    one independently -- one jersey's lock must never influence another jersey's own search."""
    take = _take(0, 0.0, 10.0)
    track_a = Track(id=100, take_id=0, boxes=[_box_at(1.0, 50, 50), _box_at(5.0, 52, 50)])
    track_c = Track(id=300, take_id=0, boxes=[_box_at(1.0, 500, 500), _box_at(5.0, 502, 500)])
    anns = [
        _annotation(1.0, jersey=10, colour="white"),
        _annotation(1.5, jersey=7, colour="red"),
        _annotation(5.0, jersey=10, colour="white"),
        _annotation(5.5, jersey=7, colour="red"),
    ]

    def fake_associate(video_path, take, candidates, ann, *args, **kwargs):
        ids = {tr.id for tr in candidates}
        agreement = {"arrow_track_id": None, "arrow_distance_px": None, "agreement": "colour_only"}
        if ann.jersey_number == 10 and 100 in ids:
            return 100, 5.0, agreement
        if ann.jersey_number == 7 and 300 in ids:
            return 300, 5.0, agreement
        raise AssertionError(f"unexpected call: jersey={ann.jersey_number} ids={ids}")

    monkeypatch.setattr(me_mod, "associate_annotation_to_track", fake_associate)

    configs = {
        "annotations": {"track_association": {}},
        "hardware": {"decode": {}},
        "highlights": {"selection": _selection_cfg()},
    }
    identity_by_take, debug = me_mod.build_manual_identity_by_take(
        [take], {0: [track_a, track_c]}, {0: anns}, "fake.mp4", configs
    )

    assert debug["0:1.00"]["track_id"] == 100
    assert debug["0:5.00"]["track_id"] == 100
    assert debug["0:1.50"]["track_id"] == 300
    assert debug["0:5.50"]["track_id"] == 300
    assert set(identity_by_take[0].location_track_ids) == {100, 300}


# ---------------------------------------------------------------------------
# run_manual_events_pipeline_for_video -- acceptance #2 wiring smoke test
# ---------------------------------------------------------------------------


def test_run_manual_events_pipeline_smoke(tmp_path, monkeypatch):
    video_path = tmp_path / "match.mp4"
    video_path.write_bytes(b"not a real video -- every I/O boundary below is mocked")
    sidecar = tmp_path / "match.annotations.txt"
    sidecar.write_text(
        "Min 0:01 player #2 in white takes a touch\n"
        "Min 0:02 player #2 in white makes a pass\n"
        "malformed line with no jersey token\n"
    )

    take = _take(0, 0.0, 10.0)
    track = _track(1, 1.0)

    class _FakeProfile:
        class profile:
            value = "single-static"

    written_players: list[dict] = []

    def fake_write_player_output(
        player_dir,
        jersey_number,
        events,
        possession_seconds,
        distance_result,
        takes_by_id,
        video_path,
        highlights_cfg,
        goal_reason=None,
        identity_status="Verified",
    ):
        written_players.append(
            {
                "jersey_number": jersey_number,
                "n_events": len(events),
                "possession_seconds": possession_seconds,
                "distance_result": distance_result,
                "goal_reason": goal_reason,
                "identity_status": identity_status,
            }
        )
        player_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(me_mod, "build_run_profile", lambda *a, **k: _FakeProfile())
    monkeypatch.setattr(me_mod, "run_track_stage", lambda *a, **k: {"n_takes": 1, "n_tracks": 1})
    monkeypatch.setattr(
        me_mod,
        "load_models_parquet",
        lambda path, model: {"takes.parquet": [take], "tracks.parquet": [track]}.get(path.name, []),
    )
    monkeypatch.setattr(me_mod, "probe", lambda path: {"width": 100, "height": 100, "fps": 30.0})
    monkeypatch.setattr(
        me_mod,
        "associate_annotation_to_track",
        lambda *a, **k: (
            1,
            5.0,
            {"arrow_track_id": None, "arrow_distance_px": None, "agreement": "colour_only"},
        ),
    )
    monkeypatch.setattr(me_mod, "render_full_annotated_video", lambda *a, **k: None)
    monkeypatch.setattr(me_mod, "write_player_output", fake_write_player_output)

    configs = {
        "annotations": load_yaml("configs/annotations.yaml"),
        "hardware": {"decode": {"scale_width": None}},
        "profile": {},
        "shots": {},
        "detect": {},
        "track": {},
        "team": {},
        "highlights": {},
    }

    result = me_mod.run_manual_events_pipeline_for_video(
        video_path,
        configs,
        sidecar,
        work_root=tmp_path / "work",
        output_root=tmp_path / "output",
        use_nvdec=False,
    )

    assert result["mode"] == "manual_events"
    assert result["jersey_numbers"] == [2]
    assert result["n_annotations_parsed"] == 2
    assert result["n_problems"] == 1

    out_dir = tmp_path / "output" / "match"
    assert (out_dir / "annotation_report.json").exists()

    assert len(written_players) == 1
    written = written_players[0]
    assert written["jersey_number"] == 2
    assert written["n_events"] == 2  # touch + pass
    assert written["possession_seconds"] is None
    assert written["distance_result"] is None
    assert written["identity_status"] == "Human-provided (manual annotation)"
    # Bug fix 2026-08-31: manual mode's sidecar is the authoritative event source, so a player
    # with no GOAL annotation has a real "0" goals, not an "uncertain"/"not available" placeholder
    # -- `goal_reason=None` unconditionally is what tells render_statcard_markdown to show that
    # real zero (see src/pipeline/player_output.py's own docstring). This is an intentional
    # behaviour change from the old pinned "not available" assertion.
    assert written["goal_reason"] is None


# ---------------------------------------------------------------------------
# Stage B ("streamed-gathering-treehouse" plan) -- jersey-number OCR/VLM re-identification,
# wired into build_manual_identity_by_take's lock/reacquisition orchestration. Both
# `associate_annotation_to_track` (colour+arrow) and `read_jersey_number_for_candidates`
# (jersey read) are monkeypatched -- no real decode/OCR/VLM -- to isolate the DECISION logic:
# which signal wins, and when reacquisition is allowed to accept a non-locked-chain candidate.
# ---------------------------------------------------------------------------


def _stub_easyocr(monkeypatch) -> None:
    """These tests exercise the LOCK/REACQUISITION decision logic with
    `read_jersey_number_for_candidates` itself monkeypatched -- the real EasyOCR reader
    (`load_easyocr_reader`) must never actually load a model here, only be created/freed as a
    cheap placeholder object."""
    monkeypatch.setattr(me_mod, "load_easyocr_reader", lambda ocr_cfg: object())
    monkeypatch.setattr(me_mod, "free_easyocr_reader", lambda reader: None)


def _jersey_reid_configs(selection_cfg: dict) -> dict:
    return {
        "annotations": {
            "track_association": {"match_tolerance_s": 0.2},
            "jersey_reid": {"enabled": True, "max_vlm_escalations_per_call": 2},
        },
        "hardware": {"decode": {}},
        "highlights": {"selection": selection_cfg},
        "identity": {"ocr": {}},
    }


def test_initial_lock_prefers_confident_jersey_read_over_colour_pick(monkeypatch):
    """A confident jersey-number read matching the annotation's own claimed number is the
    strongest signal available -- it must win over a DIFFERENT colour pick, and the disagreement
    must be logged (Golden Rule 5), never silently dropped."""
    _stub_easyocr(monkeypatch)
    take = _take(0, 0.0, 10.0)
    track_colour_pick = _track(5, 1.0)
    track_jersey_pick = _track(9, 1.0)
    anns = [_annotation(1.0, jersey=7)]

    monkeypatch.setattr(
        me_mod,
        "associate_annotation_to_track",
        lambda *a, **k: (
            5,
            10.0,
            {"arrow_track_id": None, "arrow_distance_px": None, "agreement": "colour_only"},
        ),
    )
    monkeypatch.setattr(
        me_mod,
        "read_jersey_number_for_candidates",
        lambda *a, **k: (9, "ocr", {"outcome": "matched"}),
    )

    identity_by_take, debug = me_mod.build_manual_identity_by_take(
        [take],
        {0: [track_colour_pick, track_jersey_pick]},
        {0: anns},
        "fake.mp4",
        _jersey_reid_configs(_selection_cfg()),
    )

    entry = debug["0:1.00"]
    assert entry["track_id"] == 9  # jersey read wins over colour's pick of 5
    assert entry["association_signal"] == "jersey_ocr"
    assert entry["jersey_colour_disagreement"] == {"colour_track_id": 5, "jersey_track_id": 9}
    assert identity_by_take[0].location_track_ids == [9]
    # owner-reported bug fix, 2026-08-31: a real jersey-number confirmation means the renderer's
    # panel may honestly say "VERIFIED", not just "COLOUR MATCH (jersey unconfirmed)".
    assert identity_by_take[0].association_confirmed_by_jersey is True


def test_initial_lock_falls_back_to_colour_when_jersey_read_finds_nothing(monkeypatch):
    _stub_easyocr(monkeypatch)
    take = _take(0, 0.0, 10.0)
    track = _track(5, 1.0)
    anns = [_annotation(1.0, jersey=7)]

    monkeypatch.setattr(
        me_mod,
        "associate_annotation_to_track",
        lambda *a, **k: (
            5,
            10.0,
            {"arrow_track_id": None, "arrow_distance_px": None, "agreement": "colour_only"},
        ),
    )
    monkeypatch.setattr(
        me_mod,
        "read_jersey_number_for_candidates",
        lambda *a, **k: (None, None, {"outcome": "no_match"}),
    )

    identity_by_take, debug = me_mod.build_manual_identity_by_take(
        [take],
        {0: [track]},
        {0: anns},
        "fake.mp4",
        _jersey_reid_configs(_selection_cfg()),
    )

    entry = debug["0:1.00"]
    assert entry["track_id"] == 5
    assert entry["association_signal"] == "colour"
    assert "jersey_colour_disagreement" not in entry
    # owner-reported bug fix, 2026-08-31: colour-only resolution must NOT claim "VERIFIED" --
    # the renderer needs this False to show "COLOUR MATCH (jersey unconfirmed)" instead.
    assert identity_by_take[0].association_confirmed_by_jersey is False


def test_reacquisition_accepts_jersey_confirmed_candidate_outside_locked_chain(monkeypatch):
    """Once jersey #10 locks onto track 100's chain at the first annotation, if NO member of that
    chain is visible near a LATER instant, a fresh jersey-number read confirming a DIFFERENT,
    non-locked-chain candidate (track 200) must still be accepted -- this is the honest
    re-acquisition Stage B adds, distinct from (and never substituted by) a bare colour guess."""
    _stub_easyocr(monkeypatch)
    take = _take(0, 0.0, 10.0)
    track_a = Track(id=100, take_id=0, boxes=[_box_at(1.0, 50, 50)])
    track_b = Track(id=200, take_id=0, boxes=[_box_at(5.0, 500, 500)])
    anns = [_annotation(1.0, jersey=10), _annotation(5.0, jersey=10)]

    def fake_associate(video_path, take, candidates, ann, *args, **kwargs):
        ids = {tr.id for tr in candidates}
        if ids == {100, 200} and ann.t < 3.0:
            # first annotation, no lock yet -- fresh search over the FULL take_tracks pool.
            return (
                100,
                5.0,
                {"arrow_track_id": None, "arrow_distance_px": None, "agreement": "colour_only"},
            )
        if ids == {100} and ann.t >= 3.0:
            # restricted search: locked chain (100) has no candidate near this later instant.
            return (
                None,
                None,
                {
                    "arrow_track_id": None,
                    "arrow_distance_px": None,
                    "agreement": "no_confident_signal",
                },
            )
        raise AssertionError(f"unexpected candidate id set {ids} for ann.t={ann.t}")

    def fake_jersey_read(video_path, take, candidates, t, claimed_jersey_number, *args, **kwargs):
        ids = {tr.id for tr in candidates}
        if t < 3.0:
            # first annotation: jersey read finds nothing near t=1.0 -- colour's pick (100) wins.
            assert ids == {100}
            return None, None, {"outcome": "no_match"}
        # second annotation: track_a (100) has no box near t=5.0 at all, so the re-opened
        # near-time pool is just {200} here -- the key behaviour under test is that 200 (NOT a
        # member of the locked chain) is still eligible and gets accepted purely on the
        # jersey-number match.
        assert ids == {200}
        return 200, "vlm", {"outcome": "matched"}

    monkeypatch.setattr(me_mod, "associate_annotation_to_track", fake_associate)
    monkeypatch.setattr(me_mod, "read_jersey_number_for_candidates", fake_jersey_read)

    identity_by_take, debug = me_mod.build_manual_identity_by_take(
        [take],
        {0: [track_a, track_b]},
        {0: anns},
        "fake.mp4",
        _jersey_reid_configs(_selection_cfg()),
    )

    assert debug["0:1.00"]["track_id"] == 100
    assert debug["0:5.00"]["track_id"] == 200
    assert debug["0:5.00"]["association_signal"] == "jersey_vlm_reacquired"
    assert set(identity_by_take[0].location_track_ids) == {100, 200}


def test_reacquisition_stays_caption_only_when_jersey_read_also_fails(monkeypatch):
    """When re-opening the search finds no jersey-confirmed candidate either, the honest result is
    unchanged from the pre-Stage-B behaviour: caption-only, no fabricated box."""
    _stub_easyocr(monkeypatch)
    take = _take(0, 0.0, 10.0)
    track_a = Track(id=100, take_id=0, boxes=[_box_at(1.0, 50, 50)])
    track_b = Track(id=200, take_id=0, boxes=[_box_at(5.0, 500, 500)])
    anns = [_annotation(1.0, jersey=10), _annotation(5.0, jersey=10)]

    def fake_associate(video_path, take, candidates, ann, *args, **kwargs):
        ids = {tr.id for tr in candidates}
        if ids == {100, 200} and ann.t < 3.0:
            return (
                100,
                5.0,
                {"arrow_track_id": None, "arrow_distance_px": None, "agreement": "colour_only"},
            )
        return (
            None,
            None,
            {"arrow_track_id": None, "arrow_distance_px": None, "agreement": "no_confident_signal"},
        )

    monkeypatch.setattr(me_mod, "associate_annotation_to_track", fake_associate)
    monkeypatch.setattr(
        me_mod,
        "read_jersey_number_for_candidates",
        lambda *a, **k: (None, None, {"outcome": "no_match"}),
    )

    identity_by_take, debug = me_mod.build_manual_identity_by_take(
        [take],
        {0: [track_a, track_b]},
        {0: anns},
        "fake.mp4",
        _jersey_reid_configs(_selection_cfg()),
    )

    assert debug["0:1.00"]["track_id"] == 100
    assert debug["0:5.00"]["track_id"] is None
    assert debug["0:5.00"]["agreement"] == "locked_identity_absent_at_instant"
    assert debug["0:5.00"]["association_signal"] == "none"
    assert identity_by_take[0].location_track_ids == [100]
