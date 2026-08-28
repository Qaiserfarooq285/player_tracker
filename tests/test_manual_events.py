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


def test_build_manual_identity_by_take_verified_when_association_succeeds(monkeypatch):
    take = _take(0, 0.0, 10.0)
    track = _track(1, 1.0)
    anns = [_annotation(1.0)]

    monkeypatch.setattr(me_mod, "associate_annotation_to_track", lambda *a, **k: (1, 5.0))

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


def test_build_manual_identity_by_take_no_identity_when_association_fails(monkeypatch):
    take = _take(0, 0.0, 10.0)
    track = _track(1, 1.0)
    anns = [_annotation(1.0)]

    monkeypatch.setattr(me_mod, "associate_annotation_to_track", lambda *a, **k: (None, 55.0))

    identity_by_take, _debug = me_mod.build_manual_identity_by_take(
        [take],
        {0: [track]},
        {0: anns},
        "fake.mp4",
        {"annotations": {"track_association": {}}, "hardware": {"decode": {}}},
    )
    assert identity_by_take == {}


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
    monkeypatch.setattr(me_mod, "associate_annotation_to_track", lambda *a, **k: (1, 5.0))
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
    assert written["goal_reason"] is not None  # no GOAL annotation -> explicit reason, never blank
    assert "not available" in written["goal_reason"]
