"""Stage E of the "streamed-gathering-treehouse" plan (`apps/api/main.py`): "the click is a
candidate, not a command." Pure function-level unit tests with everything I/O-heavy (video decode,
jersey OCR/PARSeq model loading, parquet reads) mocked out -- no GPU, no real video, no network,
mirroring `tests/test_target_verify.py`'s own real-measured-fixture style for the one REJECT case
that exercises the genuine `verify_candidate` scoring path end to end.

Covers:
- `ProcessRequest.target_jersey` now defaults to `None`, not `10` (the confirmed real
  `chelsea_burnley_target2` contamination bug).
- `_find_output_dir` is EXACT match only, never a substring fallback.
- `_resolve_target_click`: no stored profile establishes TARGET_001 (persisted); a stored profile
  treats the click as a candidate -- ACCEPT reconnects (same `target_id`, no redundant
  save/rebuild here -- that is `run_pipeline_for_video`'s own job, see that function's docstring),
  REJECT/UNCERTAIN raises `ClickRejected` and never touches the stored profile.
"""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.common.io import load_yaml
from src.common.types import BBox, Track, TrackBox
from src.track.target import KitColourSample, TargetProfile
from src.track.target_verify import TargetVerdict, VerdictDecision

main = importlib.import_module("apps.api.main")

REPO_ROOT = Path(__file__).resolve().parents[1]

# Same real, measured CIELAB values as tests/test_target_verify.py (chelsea_burnley_target10,
# 2026-09-02) -- reused here, not re-derived, so the one real (non-mocked) REJECT test below
# exercises `verify_candidate`'s actual scoring against the exact real failure this whole plan
# exists to catch, not a synthetic pair.
BLUE_10 = (44.9, 7.5, -24.0)  # Chelsea #10, visually confirmed
CLARET_21 = (52.9, 5.0, -1.0)  # Burnley #21, visually confirmed -- dE from BLUE_10 is 24.5


def _cfg() -> dict:
    """The two config sections `_resolve_target_click` actually reads -- `configs["target"]` (for
    `build_target_profile`) and `configs["events"]["kit_colour"]` (merged into `target_cfg` for
    `verify_candidate`, exactly as that module's own docstring requires). `_click_evidence_for_
    candidate` is mocked out in every test below, so it never needs the other config sections a
    real `_run_pipeline_job` call would supply."""
    return {
        "target": load_yaml(REPO_ROOT / "configs" / "target.yaml"),
        "events": load_yaml(REPO_ROOT / "configs" / "events.yaml"),
    }


def _box(t: float, height: float = 100.0) -> TrackBox:
    return TrackBox(frame_index=int(t * 10), t=t, bbox=BBox(x1=0, y1=0, x2=50, y2=height), conf=0.9)


def _track(tid: int, boxes: list[TrackBox], team: int | None = None) -> Track:
    return Track(id=tid, take_id=0, boxes=boxes, team=team)


def _kit(torso=None, shorts=None, socks=None) -> KitColourSample:
    return KitColourSample(
        torso_lab=torso, shorts_lab=shorts, socks_lab=socks, take_id=0, t=1.0, confidence=1.0
    )


def _profile(
    kit: KitColourSample | None = None,
    jersey_number: int | None = None,
    median_height: float = 100.0,
) -> TargetProfile:
    return TargetProfile(
        jersey_number=jersey_number,
        jersey_source="click" if jersey_number is not None else None,
        kit=kit,
        kit_bank=[kit] if kit is not None else [],
        median_height=median_height,
        team_cluster=0,
        established_take_id=0,
        established_track_id=1,
        established_t=0.0,
        links=[],
    )


# ---------------------------------------------------------------------------
# ProcessRequest.target_jersey default
# ---------------------------------------------------------------------------


def test_process_request_target_jersey_defaults_to_none():
    """The confirmed real bug: a request that omits `target_jersey` entirely (a pure click/
    track-id selection with no jersey field filled in) must NOT silently become jersey #10 --
    `work/chelsea_burnley_target2/selection.json` carrying `target_jersey: 10` for a video whose
    own filename says `target2` is exactly this default leaking through."""
    req = main.ProcessRequest(video_name="video3/chelsea_burnley_target2.mp4")
    assert req.target_jersey is None


def test_process_request_target_jersey_still_honours_an_explicit_value():
    req = main.ProcessRequest(video_name="clip2 77.mp4", target_jersey=77)
    assert req.target_jersey == 77


# ---------------------------------------------------------------------------
# `ProcessRequest.target_clicks` / `TargetAnchor` -- "streamed-gathering-treehouse" plan Stage 1
# ---------------------------------------------------------------------------


def test_process_request_target_clicks_defaults_to_none():
    """Omitting the field entirely (every pre-existing request shape: raw-coordinate click, or a
    hand-typed `track_id`) must not implicitly become an empty list -- `None` and `[]` are
    distinguishable states the merge logic in `_run_pipeline_job` can tell apart."""
    req = main.ProcessRequest(video_name="x.mp4")
    assert req.target_clicks is None


def test_process_request_accepts_multiple_target_click_anchors():
    """The picker's own accumulating chip list submits one `TargetAnchor` per click, possibly
    several in the SAME take (the "player left frame, came back with a new track id" case)."""
    req = main.ProcessRequest(
        video_name="x.mp4",
        target_clicks=[
            {"take_id": 0, "track_id": 44, "t": 0.5},
            {"take_id": 0, "track_id": 51, "t": 12.3},
            {"take_id": 1, "track_id": 9},
        ],
    )
    assert req.target_clicks == [
        main.TargetAnchor(take_id=0, track_id=44, t=0.5),
        main.TargetAnchor(take_id=0, track_id=51, t=12.3),
        main.TargetAnchor(take_id=1, track_id=9, t=None),
    ]


def test_target_anchor_t_is_optional():
    anchor = main.TargetAnchor(take_id=0, track_id=16)
    assert anchor.t is None


# ---------------------------------------------------------------------------
# /api/results slug matching -- exact only
# ---------------------------------------------------------------------------


def test_find_output_dir_exact_match(tmp_path):
    (tmp_path / "chelsea_burnley_target10").mkdir()
    found = main._find_output_dir(tmp_path, "chelsea_burnley_target10")
    assert found == tmp_path / "chelsea_burnley_target10"


def test_find_output_dir_rejects_substring_match(tmp_path):
    """Real bug fixed here: the old `slug in d.name or d.name in slug` fallback would have
    matched `chelsea_burnley_target1` against the on-disk `chelsea_burnley_target10` (the shorter
    string is a substring of the longer one) -- serving a DIFFERENT run's results. Exact match
    only means this must now come back `None`, not the wrong directory."""
    (tmp_path / "chelsea_burnley_target10").mkdir()
    assert main._find_output_dir(tmp_path, "chelsea_burnley_target1") is None
    assert main._find_output_dir(tmp_path, "chelsea_burnley_target100") is None


def test_find_output_dir_no_match_returns_none(tmp_path):
    assert main._find_output_dir(tmp_path, "nothing_here") is None


# ---------------------------------------------------------------------------
# _resolve_target_click
# ---------------------------------------------------------------------------


def test_resolve_target_click_first_click_establishes_and_persists(tmp_path, monkeypatch):
    """No stored profile -> this selection ESTABLISHES `TARGET_001`, authoritative, and is
    persisted immediately (`save_target_profile`) -- plan Stage E's first bullet."""
    monkeypatch.setattr(main, "WORK_DIR", tmp_path)  # never touch the real repo work/ tree

    candidate = _track(16, [_box(0.0, 100), _box(1.0, 100)], team=0)
    kit_sample = _kit(torso=BLUE_10)
    monkeypatch.setattr(
        main, "_click_evidence_for_candidate", lambda *a, **k: (candidate, kit_sample, "10")
    )

    saved: dict = {}

    def _fake_save(profile, path):
        saved["profile"] = profile
        saved["path"] = path

    monkeypatch.setattr("src.track.target.save_target_profile", _fake_save)

    result = main._resolve_target_click(
        Path("input/fake_video.mp4"), _cfg(), None, 0, 16, click_t=12.5
    )

    assert isinstance(result, TargetProfile)
    assert result.target_id == "TARGET_001"
    assert result.jersey_source == "click"
    assert result.jersey_number == 10
    assert result.established_take_id == 0
    assert result.established_track_id == 16
    assert result.established_t == 12.5
    assert result.kit is not None  # the one supplied kit sample aggregated in
    # persisted immediately, not deferred to the (out-of-scope) pipeline run:
    assert saved["profile"] is result
    assert saved["path"] == tmp_path / "fake_video" / "target.json"


def test_resolve_target_click_missing_candidate_is_a_loud_failure(monkeypatch):
    """Defensive branch: `_click_evidence_for_candidate` returning `(None, None, None)` (the
    track could not even be re-located) must raise, never silently proceed."""
    monkeypatch.setattr(main, "_click_evidence_for_candidate", lambda *a, **k: (None, None, None))
    with pytest.raises(main.ClickRejected):
        main._resolve_target_click(Path("input/fake_video.mp4"), _cfg(), None, 0, 99, None)


def test_resolve_target_click_second_click_accept_reconnects_same_target(monkeypatch):
    """A stored profile exists -> ACCEPT reconnects: the SAME profile object/`target_id` comes
    back, and -- per `_resolve_target_click`'s own docstring -- neither `build_target_profile` nor
    `save_target_profile` is called here at all: the actual reconnect + memory-bank update +
    persistence is `run_pipeline_for_video`'s own already-wired job when it re-verifies this same
    override a moment later, so doing it twice here would double-insert the same evidence."""
    profile = _profile(kit=_kit(torso=BLUE_10), jersey_number=10)
    candidate = _track(52, [_box(30.0, 100)], team=0)
    monkeypatch.setattr(
        main,
        "_click_evidence_for_candidate",
        lambda *a, **k: (candidate, _kit(torso=BLUE_10), "10"),
    )

    fake_verdict = TargetVerdict(decision=VerdictDecision.ACCEPT, score=0.95, evidence={})
    verify_mock = MagicMock(return_value=fake_verdict)
    monkeypatch.setattr("src.track.target_verify.verify_candidate", verify_mock)

    build_mock = MagicMock()
    save_mock = MagicMock()
    monkeypatch.setattr("src.track.target.build_target_profile", build_mock)
    monkeypatch.setattr("src.track.target.save_target_profile", save_mock)

    result = main._resolve_target_click(
        Path("input/fake_video.mp4"), _cfg(), profile, 1, 52, click_t=None
    )

    assert result is profile  # identically the SAME TargetProfile / target_id, a true reconnect
    verify_mock.assert_called_once()
    build_mock.assert_not_called()
    save_mock.assert_not_called()


def test_resolve_target_click_second_click_reject_keeps_target_lost(monkeypatch):
    """The real measured failure this whole plan exists to catch, run through `_resolve_target_
    click` end to end with the REAL `verify_candidate` (not mocked): a profile built on Chelsea's
    blue #10 must REJECT a claret Burnley #21 candidate on kit colour alone. The stored profile
    must never be touched (`build_target_profile`/`save_target_profile` both unused) and the
    raised message must name the reason in plain language."""
    profile = _profile(kit=_kit(torso=BLUE_10, shorts=BLUE_10, socks=BLUE_10), jersey_number=10)
    candidate = _track(47, [_box(40.0, 100)], team=0)
    claret_kit = _kit(torso=CLARET_21, shorts=CLARET_21, socks=CLARET_21)
    monkeypatch.setattr(
        main, "_click_evidence_for_candidate", lambda *a, **k: (candidate, claret_kit, None)
    )

    build_mock = MagicMock()
    save_mock = MagicMock()
    monkeypatch.setattr("src.track.target.build_target_profile", build_mock)
    monkeypatch.setattr("src.track.target.save_target_profile", save_mock)

    with pytest.raises(main.ClickRejected) as excinfo:
        main._resolve_target_click(Path("input/fake_video.mp4"), _cfg(), profile, 0, 47, None)

    message = str(excinfo.value)
    assert "CLICK REJECTED" in message
    assert "TARGET_001" in message
    assert "different kit colour" in message
    assert "Target remains lost" in message
    build_mock.assert_not_called()
    save_mock.assert_not_called()


def test_resolve_target_click_uncertain_no_longer_rejects_a_human_click(monkeypatch):
    """Owner-reported failure, 2026-09-15: a correct reconnect click was refused with
    `CLICK REJECTED ... (match uncertain, score=0.85)` while EVERY hard check had passed.

    A human click is now governed by the hard rejects (kit colour / jersey number / height), not by
    the blended soft score. The blend's `trajectory` term rewards continuity with where the target
    was last seen -- and a reconnect click happens precisely because the target vanished and
    reappeared elsewhere, so scoring that discontinuity is circular. CLAUDE.md Golden Rule 4: a
    human-provided identity is the strongest evidence this pipeline has.

    (Automatic re-identification keeps the strict bar -- see
    `src.highlights.selection._select_take_with_profile`; there no human asserted anything.)"""
    profile = _profile(kit=_kit(torso=BLUE_10), jersey_number=10)
    candidate = _track(9, [_box(5.0, 100)], team=0)
    monkeypatch.setattr(
        main,
        "_click_evidence_for_candidate",
        lambda *a, **k: (candidate, _kit(torso=BLUE_10), "10"),
    )
    fake_verdict = TargetVerdict(decision=VerdictDecision.UNCERTAIN, score=0.85, evidence={})
    monkeypatch.setattr(
        "src.track.target_verify.verify_candidate", MagicMock(return_value=fake_verdict)
    )

    returned = main._resolve_target_click(Path("input/fake_video.mp4"), _cfg(), profile, 0, 9, None)
    assert returned is profile  # reconnected, same persistent identity, nothing overwritten


@pytest.mark.parametrize(
    "reason_code", ["wrong_kit_colour", "wrong_jersey_number", "height_mismatch"]
)
def test_resolve_target_click_still_rejects_on_a_hard_signal(monkeypatch, reason_code):
    """The owner's own requirement is unchanged: clicking a genuinely DIFFERENT player is refused
    and the target stays lost. Those three hard signals are the reliable discriminators, and they
    still veto a click outright -- only the soft-score veto was removed."""
    profile = _profile(kit=_kit(torso=BLUE_10), jersey_number=10)
    candidate = _track(9, [_box(5.0, 100)], team=0)
    monkeypatch.setattr(
        main,
        "_click_evidence_for_candidate",
        lambda *a, **k: (candidate, _kit(torso=BLUE_10), "10"),
    )
    fake_verdict = TargetVerdict(
        decision=VerdictDecision.REJECT, score=0.0, evidence={"rejected_reason": reason_code}
    )
    monkeypatch.setattr(
        "src.track.target_verify.verify_candidate", MagicMock(return_value=fake_verdict)
    )
    save_mock = MagicMock()
    monkeypatch.setattr("src.track.target.save_target_profile", save_mock)

    with pytest.raises(main.ClickRejected) as excinfo:
        main._resolve_target_click(Path("input/fake_video.mp4"), _cfg(), profile, 0, 9, None)
    assert "CLICK REJECTED" in str(excinfo.value)
    save_mock.assert_not_called()  # a rejected click never touches the stored profile


def test_resolve_target_click_never_rejects_the_establishing_anchor(monkeypatch):
    """Re-clicking the exact track that ESTABLISHED this profile must never be told it "does not
    match" itself. `src.highlights.selection._select_take_with_profile` already had this bypass;
    this path did not, so the owner could be rejected by their own anchor."""
    profile = _profile(kit=_kit(torso=BLUE_10), jersey_number=10)
    profile.established_track_id = 9
    profile.established_take_id = 0
    candidate = _track(9, [_box(5.0, 100)], team=0)
    monkeypatch.setattr(
        main,
        "_click_evidence_for_candidate",
        lambda *a, **k: (candidate, _kit(torso=BLUE_10), "10"),
    )
    verify_mock = MagicMock()
    monkeypatch.setattr("src.track.target_verify.verify_candidate", verify_mock)

    returned = main._resolve_target_click(Path("input/fake_video.mp4"), _cfg(), profile, 0, 9, None)
    assert returned is profile
    verify_mock.assert_not_called()  # the anchor is the target by definition, not by score


# ---------------------------------------------------------------------------
# _click_rejection_message
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reason_code,expected_label",
    [
        ("wrong_kit_colour", "different kit colour"),
        ("wrong_jersey_number", "different jersey number"),
        ("height_mismatch", "different player height"),
        ("score_below_uncertain_low", "overall match score too low"),
    ],
)
def test_click_rejection_message_labels_every_hard_reject_reason(reason_code, expected_label):
    profile = _profile(jersey_number=11)
    verdict = TargetVerdict(
        decision=VerdictDecision.REJECT, score=0.0, evidence={"rejected_reason": reason_code}
    )
    message = main._click_rejection_message(profile, verdict)
    assert expected_label in message
    assert "#11" in message
    assert "TARGET_001" in message


# ---------------------------------------------------------------------------
# Plan Fix C ("streamed-gathering-treehouse" re-check, 2026-09-15): "the click is a candidate, not
# a command" extends to the FRONTEND too -- `apps/web/js/app.js`'s raw-coordinate click handler
# (`initClickToTrack`) used to treat every click on the video preview as a fresh single-candidate
# selection and unconditionally run `targetAnchors = [];`, so the owner's own stated workflow
# (click the player, seek forward -- by clicking the video -- click again, seek again...) silently
# lost every anchor but the last one clicked. This repo has no JS test runner/package.json (no
# jsdom dependency either), so this drives the REAL, unmodified `app.js` source directly via
# Node's built-in `vm` module inside a minimal hand-built DOM stub
# (`tests/js/click_anchor_harness.js`) -- skipped, never failed, when `node` isn't on PATH.
# ---------------------------------------------------------------------------

REPO_ROOT_FOR_JS = Path(__file__).resolve().parents[1]


def _run_click_anchor_harness() -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed -- cannot exercise the real app.js source")
    harness = REPO_ROOT_FOR_JS / "tests" / "js" / "click_anchor_harness.js"
    app_js = REPO_ROOT_FOR_JS / "apps" / "web" / "js" / "app.js"
    proc = subprocess.run(
        [node, str(harness), str(app_js)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"harness failed: stderr={proc.stderr!r}"
    return json.loads(proc.stdout)


def test_raw_click_still_pins_a_candidate_when_no_anchors_exist():
    """Backwards compatibility (explicitly required by the plan): with the chip list empty, a
    raw coordinate click on the video still behaves exactly as it did before Fix C -- pins
    `selectedClickPoint` and un-hides the marker."""
    results = _run_click_anchor_harness()
    assert results["emptyListPinsCandidate"] is True
    assert results["markerVisibleAfterFirstClick"] is True


def test_raw_click_no_longer_empties_anchor_array_once_anchors_exist():
    """THE regression test for the real bug: once the picker has accumulated anchors, clicking
    the video preview again (the owner's own "seek to the next moment" action) must NOT wipe
    `targetAnchors` -- both its length and its exact contents must survive untouched, and the
    chip list must stay visible. Confirmed this harness actually catches the pre-fix bug (not just
    a tautology) by running it against the pre-Fix-C `app.js` from git history: there,
    `anchorCountAfterVideoClick` drops to `0` and `chipsStillVisibleAfterVideoClick` is `False`."""
    results = _run_click_anchor_harness()
    assert results["anchorCountBeforeVideoClick"] == 2
    assert results["chipsVisibleBeforeVideoClick"] is True
    assert results["anchorCountAfterVideoClick"] == 2
    assert results["chipsStillVisibleAfterVideoClick"] is True
    assert results["anchorsUnchangedAfterVideoClick"] is True
