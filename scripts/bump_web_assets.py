"""Rewrite the `?v=<hash>` cache-busters in apps/web/index.html from the current contents of
css/style.css and js/app.js. Run after editing either file (tests/test_web_assets.py fails when
the hashes are stale, so a forgotten bump never ships a page whose browser-cached CSS/JS is old).

    python scripts/bump_web_assets.py
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "apps" / "web"
ASSETS = ("css/style.css", "js/app.js")


def content_hash(rel: str) -> str:
    return hashlib.sha1((WEB / rel).read_bytes()).hexdigest()[:8]


def expected_and_actual() -> list[tuple[str, str, str | None]]:
    """`(asset, expected_hash, hash_in_index)` per asset."""
    html = (WEB / "index.html").read_text()
    out = []
    for rel in ASSETS:
        m = re.search(re.escape(rel) + r"\?v=([0-9a-f]+)", html)
        out.append((rel, content_hash(rel), m.group(1) if m else None))
    return out


def bump() -> bool:
    index = WEB / "index.html"
    html = index.read_text()
    changed = False
    for rel, expected, actual in expected_and_actual():
        if actual == expected:
            continue
        html = re.sub(re.escape(rel) + r"\?v=[0-9a-f]+", f"{rel}?v={expected}", html)
        print(f"{rel}: {actual} -> {expected}")
        changed = True
    if changed:
        index.write_text(html)
    return changed


if __name__ == "__main__":
    sys.exit(0 if bump() or True else 1)
