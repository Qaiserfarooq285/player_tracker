"""ffmpeg encode plumbing in `src/common/video.py`: the NVENC probe and the stdin frame pipe.
Owner-visible failure this covers (2026-09-17, RunPod RTX 4090 pod): the annotated render died
with a bare `[Errno 32] Broken pipe` because NVENC is not exposed in the container -- ffmpeg
exited on frame 0, the BrokenPipeError pre-empted the returncode/stderr check, and the caller's
libx264 fallback (which catches RuntimeError) never ran. No real ffmpeg is launched here."""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

import src.common.video as video


class _DeadStdin:
    """ffmpeg that has already exited: every write is a broken pipe, so is close()."""

    def write(self, _data):
        raise BrokenPipeError(32, "Broken pipe")

    def close(self):
        raise BrokenPipeError(32, "Broken pipe")


class _FakeProc:
    def __init__(self, stdin, returncode):
        self.stdin = stdin
        self.returncode = returncode
        self.waited = False

    def wait(self):
        self.waited = True


def _frames(n):
    for _ in range(n):
        yield np.zeros((2, 2, 3), dtype=np.uint8)


def test_pipe_frames_swallows_broken_pipe_and_still_waits_for_the_process():
    proc = _FakeProc(_DeadStdin(), returncode=1)
    n = video.pipe_frames(proc, _frames(5))  # must NOT raise BrokenPipeError
    assert n == 0
    assert proc.waited is True
    assert proc.returncode == 1  # the caller's own check now gets to explain the failure


def test_pipe_frames_counts_frames_on_the_happy_path():
    class _OkStdin:
        def __init__(self):
            self.bytes = 0

        def write(self, data):
            self.bytes += len(data)

        def close(self):
            pass

    stdin = _OkStdin()
    proc = _FakeProc(stdin, returncode=0)
    assert video.pipe_frames(proc, _frames(3)) == 3
    assert stdin.bytes == 3 * 2 * 2 * 3


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    video.nvenc_available.cache_clear()
    yield
    if hasattr(video.nvenc_available, "cache_clear"):  # a test may have monkeypatched it away
        video.nvenc_available.cache_clear()


def test_nvenc_probe_false_when_encoder_cannot_open_a_session(monkeypatch):
    calls = []

    def fake_run(cmd, capture_output=None, timeout=None):
        calls.append(cmd)
        return subprocess.CompletedProcess(
            cmd, 1, stdout=b"",
            stderr=b"[h264_nvenc @ 0x1] OpenEncodeSessionEx failed: unsupported device (2): (no details)\n",
        )

    monkeypatch.setattr(video.subprocess, "run", fake_run)
    assert video.nvenc_available() is False
    assert video.nvenc_available() is False  # cached: one probe per process
    assert len(calls) == 1
    assert "h264_nvenc" in calls[0] and "-frames:v" in calls[0]


def test_nvenc_probe_true_on_success_and_false_when_ffmpeg_is_missing(monkeypatch):
    monkeypatch.setattr(video.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, b"", b""))
    assert video.nvenc_available() is True
    video.nvenc_available.cache_clear()

    def missing(cmd, **kw):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(video.subprocess, "run", missing)
    assert video.nvenc_available() is False


def test_extract_clip_skips_nvenc_when_probe_says_no(monkeypatch, tmp_path):
    monkeypatch.setattr(video, "nvenc_available", lambda: False)
    seen = []

    def fake_run(cmd, capture_output=None, text=None):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(video.subprocess, "run", fake_run)
    video.extract_clip(tmp_path / "in.mp4", tmp_path / "out.mp4", 0.0, 1.0)
    assert len(seen) == 1 and "libx264" in seen[0] and "h264_nvenc" not in seen[0]
