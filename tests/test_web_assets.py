"""The cache-busters in apps/web/index.html must match the shipped css/js (see
scripts/bump_web_assets.py) -- otherwise a redeploy leaves browsers on the old stylesheet."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from bump_web_assets import expected_and_actual  # noqa: E402


def test_index_html_cache_busters_match_asset_contents():
    stale = [(rel, exp, act) for rel, exp, act in expected_and_actual() if exp != act]
    assert not stale, f"stale ?v= hashes {stale}: run `python scripts/bump_web_assets.py`"
