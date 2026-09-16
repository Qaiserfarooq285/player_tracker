"""Access-gate password resolution (docs/DEPLOY.md "Login").

Shared by `apps/api/main.py` (enforces the gate on every request) and `start_app.py` (prints the
startup banner) so the "what counts as unset / off" rule lives in exactly one place rather than
being reimplemented twice (CLAUDE.md §10: no duplicated magic values).

The gate is **on by default** (owner, 2026-09-16): a hosted pod that forgets to set
`PV_ACCESS_PASSWORD` should still require a login, not sit open to anyone with the URL. The
built-in default password is meant to be changed once the pod is up; `PV_ACCESS_PASSWORD=off`
exists only to turn the gate off for local editing.
"""

from __future__ import annotations

DEFAULT_ACCESS_PASSWORD = "admin1122"
_DISABLE_SENTINEL = "off"


def resolve_access_password(raw: str | None) -> tuple[str, bool]:
    """Resolve the raw `PV_ACCESS_PASSWORD` env value into `(password, is_default)`.

    - `None` / empty / whitespace-only -> treated as unset -> the built-in default,
      `is_default=True`.
    - the literal `off` (case-insensitive, after stripping) -> gate disabled -> `("", False)`.
    - anything else -> that value verbatim (stripped) -> `is_default=False`.
    """
    stripped = (raw or "").strip()
    if not stripped:
        return DEFAULT_ACCESS_PASSWORD, True
    if stripped.lower() == _DISABLE_SENTINEL:
        return "", False
    return stripped, False
