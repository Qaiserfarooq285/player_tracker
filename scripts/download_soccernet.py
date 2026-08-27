"""Pull one SoccerNet broadcast clip into `input/` for testing ADR-15's jersey-verification
pipeline on professional-broadcast footage (§9.1: SoccerNet is our labeled/eval-quality data
source; §7: `SoccerNet` PyPI package is MIT, the *data* is research/education-only until the NDA
is signed).

**Requires an NDA password.** SoccerNet's game videos are copyrighted broadcast footage — the
maintainers gate them behind a Non-Disclosure Agreement (a short Google Form at
https://www.soccer-net.org, tied to the requester's own identity/email, not something this
pipeline can complete on the owner's behalf). Listing available match names does NOT need the
password (the game-list JSON ships inside the `SoccerNet` pip package); only the actual video
file download does. Put the password in `.env` as `SOCCERNET_PASSWORD` once you have it.

Usage::

    .venv/bin/python scripts/download_soccernet.py --list                    # no password needed
    .venv/bin/python scripts/download_soccernet.py                          # downloads the default match
    .venv/bin/python scripts/download_soccernet.py --game "england_epl/2016-2017/2016-09-24 - 14-30 Manchester United 4 - 1 Leicester"
    .venv/bin/python scripts/download_soccernet.py --trim-seconds 0         # keep the full half (large: ~1GB+)

By default this trims the downloaded half down to the first `--trim-seconds` (300s = 5 min) via
a lossless `ffmpeg -c copy` cut before placing it in `input/` — CLAUDE.md §6 says "validate on a
short clip first," and a full 45-minute broadcast half is unnecessary just to check whether
jersey numbers are legible to OCR/VLM on this footage. Pass `--trim-seconds 0` to keep the whole
downloaded half instead.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.logging import get_logger  # noqa: E402

logger = get_logger("download_soccernet")
app = typer.Typer(add_completion=False)

# A big, well-lit, top-flight broadcast match -- picked from SoccerNet's own bundled test-split
# game list (no NDA needed to see this list, see module docstring) specifically because EPL
# top-flight broadcasts use tight, players-fill-the-frame camera work, the opposite end of the
# legibility spectrum from the wide-fixed-camera youth Veo footage in §3.1/§3.3.
DEFAULT_GAME = "england_epl/2016-2017/2016-09-24 - 14-30 Manchester United 4 - 1 Leicester"
DEFAULT_FILE = "1_720p.mkv"  # first half, 720p -- HD video needs the NDA password either way
# (224p is also NDA-gated, just lower quality; there is no free/preview tier, see CLAUDE.md ADR
# discussion / the SoccerNet FAQ).

INPUT_DIR = Path("input")


@app.command()
def main(
    game: str = typer.Option(DEFAULT_GAME, help="SoccerNet game path, '<league>/<season>/<match>'"),
    file: str = typer.Option(DEFAULT_FILE, help="Video file within the game, e.g. 1_720p.mkv"),
    trim_seconds: int = typer.Option(
        300, help="Keep only the first N seconds (lossless cut). 0 = keep the full file."
    ),
    list_games: bool = typer.Option(
        False, "--list", help="List a sample of available game paths per league and exit (no password needed)."
    ),
) -> None:
    load_dotenv()

    from SoccerNet.Downloader import SoccerNetDownloader
    from SoccerNet.utils import getListGames

    if list_games:
        games = getListGames("test", task="spotting")
        logger.info("%d games in the SoccerNet 'test' split (spotting task):", len(games))
        for g in games[:25]:
            print(g)
        if len(games) > 25:
            print(f"... and {len(games) - 25} more")
        return

    import os

    password = os.environ.get("SOCCERNET_PASSWORD", "")
    if not password:
        logger.error(
            "SOCCERNET_PASSWORD is not set in .env. SoccerNet's video files are NDA-gated -- "
            "fill in the form at https://www.soccer-net.org, wait for the password by email "
            "(check spam), then add SOCCERNET_PASSWORD=<password> to .env and re-run. "
            "(--list works without a password.)"
        )
        raise typer.Exit(code=1)

    work_root = Path("work/_soccernet_raw")
    work_root.mkdir(parents=True, exist_ok=True)

    downloader = SoccerNetDownloader(LocalDirectory=str(work_root))
    downloader.password = password

    logger.info("downloading %s / %s (this is a real broadcast video, can take a while)...", game, file)
    downloader.downloadGame(game=game, files=[file], spl="test")

    downloaded_path = work_root / game / file
    if not downloaded_path.exists():
        logger.error(
            "download did not produce %s -- check the password is correct and the game/file "
            "names match a real SoccerNet entry (--list to check available games).",
            downloaded_path,
        )
        raise typer.Exit(code=1)

    slug = "soccernet_" + game.split("/")[-1].lower().replace(" ", "_").replace("-", "")
    slug = "".join(c for c in slug if c.isalnum() or c == "_")[:80]
    dest = INPUT_DIR / f"{slug}.mp4"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.warning("removing existing %s before writing the new download", dest)
        dest.unlink()

    if trim_seconds > 0:
        logger.info("trimming to the first %ds (lossless -c copy) -> %s", trim_seconds, dest)
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y",
                "-i", str(downloaded_path),
                "-t", str(trim_seconds),
                "-c", "copy",
                str(dest),
            ],
            check=True,
        )
    else:
        logger.info("copying full file (no trim) -> %s", dest)
        shutil.copy2(downloaded_path, dest)

    logger.info(
        "done: %s ready in input/. No filename jersey number is encoded (matches ADR-15's "
        "filename-less path) -- target_jersey will be None and the verified-jersey pipeline "
        "runs the same way it did for jordan_thomas_highlight_video, but note this clip has NO "
        "burned-in arrow: the location prior falls straight to heuristic_fallback_seed, so "
        "whichever player track that heuristic locks onto is what gets jersey-checked -- this "
        "validates the OCR/VLM legibility question, not a specific chosen target.",
        dest,
    )


if __name__ == "__main__":
    app()
