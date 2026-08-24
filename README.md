# AI Soccer Highlight Analyzer

Ingests a full broadcast soccer match (or, right now, short 4K single-camera clips — see
`CLAUDE.md` §3.1) and produces **per-player highlight reels** + a **basic stat card**.

`Upload match → choose a player → process → review actions → auto/select highlights → export reel`

Built as a **pipeline of specialized, pretrained models** (never trained from scratch), stage by
stage, with every intermediate artifact cached to disk so runs are resumable. See `CLAUDE.md` for
the full rationale, architecture, and conventions — **it is the source of truth for this project;
read it before changing anything here.**

## Current phase

**Phase 1 — within-take core with MANUAL player selection.** No jersey auto-ID or fine-grained
action stats yet (those land in Phase 2+). See `CLAUDE.md` §3 and §6.

## Quickstart

```bash
export PATH="$HOME/.local/bin:$PATH"   # uv
make setup                              # create the venv, install core + dev deps
cp .env.example .env                    # fill in ROBOFLOW_API_KEY (see .env.example)
# drop a video into input/, named like: clip1 43.mp4  (jersey number is parsed from the filename)
make run                                # process input/ -> output/<clip>_<jersey>/
make test                               # unit tests
make eval                               # detection mAP + tracking HOTA on data/eval/
```

GPU extras (RF-DETR/torch, team-embedding stack) are installed separately via `make install-gpu`
once you're ready to run real inference — they are not part of the base `make setup`.

## I/O contract (CLAUDE.md §10)

- `input/` — source video(s), auto-detected. Filename convention: `clip<N> <jersey>.mp4`.
- `work/` — cached, resumable per-stage artifacts (parquet/JSON + overlay video), chunked.
- `output/<clip>_<jersey>/` — final reel, stat card, and run report for one processed video.
- `configs/*.yaml` — one file per pipeline stage; **no magic numbers in code**, every tunable lives
  here.
- `data/eval/` — labeled slices for the evaluation harness (`make eval`).

## Data contracts

All stages compose through the pydantic v2 models in `src/common/types.py` (`Frame`, `Detection`,
`BallDetection`, `PitchHomography`, `Take`, `Track`, `Event`, `Clip`, `PlayerStats`, `RunProfile`,
`RunReport`, ...). No stage invents a field it can't justify — unknowns are explicit (`None` +
a confidence), never guessed. See `CLAUDE.md` §4.

## License hygiene

This project ships **no AGPL dependencies**. In particular: **RF-DETR, not Ultralytics YOLO;
`supervision`'s ByteTrack, not BoxMOT** — both YOLO and BoxMOT are AGPL-3.0 and are never added to
any dependency list here (Golden Rule 6). See `CLAUDE.md` §7 for the full license table, checked
before every new dependency is added.

## Source of truth

**`CLAUDE.md`** — architecture, phases, golden rules, hardware budget, and the decision log
(ADRs). Update it in the same commit as any decision it documents.
