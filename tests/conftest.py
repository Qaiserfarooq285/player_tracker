"""Test-wide fixtures.

The access gate (`apps.api.main.ACCESS_PASSWORD`) is ON by default (`apps/api/access.py`,
2026-09-16) so a hosted pod is never open to the public by accident -- but most of the existing
API tests exercise endpoints without logging in first; they were written when "no
`PV_ACCESS_PASSWORD` set" meant the gate was off. Rather than touch every one of those test files,
the gate is forced off here for every test, via the standard `monkeypatch` fixture (undone
automatically after each test). `tests/test_api_upload_auth.py`'s own gate tests re-enable it
explicitly per test with the SAME `monkeypatch` instance, which layers on top of (and simply
overrides) this fixture's `setattr` within that one test -- both are unwound together afterwards.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _access_gate_off_by_default(monkeypatch):
    try:
        import apps.api.main as api_main
    except Exception:
        # apps.api.main pulls in FastAPI/uvicorn; a test file that never imports it shouldn't fail
        # here just because e.g. an optional extra isn't installed in this environment.
        return
    monkeypatch.setattr(api_main, "ACCESS_PASSWORD", "")
