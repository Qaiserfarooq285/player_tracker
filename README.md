# AI Soccer Highlight Analyzer

Give it **any clip and a target jersey number**, and it produces, fully automatically:

1. an **annotated copy of the same video** — the target player boxed and labelled with their **jersey number**,
   with the **ball tracked** throughout;
2. a **stat card** with a **timestamped event timeline** (every touch, pass, sprint, shot, tackle, save, goal,
   assist, …); and
3. **per-category highlight reels** — a passes reel, a goals reel, an assists reel, and more — one folder per
   player number.

`Upload video → (jersey # from filename, flag, auto-read, or manual annotations) → process → annotated video + stat card + highlight reels`

When footage is clear, it reads the jersey number and tracks automatically (SoccerNet-trained detector + OCR/VLM
identity). **When footage is too poor to read, the client supplies a short text file of timestamped annotations
and the pipeline runs from those instead** — no guessing, ever.

Built as a **pipeline of specialized, pretrained models** (never trained from scratch), stage by stage, with
every intermediate artifact cached to disk so runs are resumable. See `CLAUDE.md` for the full rationale,
architecture, and conventions — **it is the source of truth for this project; read it before changing anything
here.**

## Current phase

**Phase 1 — within-take core**, plus owner-authorized pull-forwards: verified jersey auto-ID for legible,
filename-less inputs (`CLAUDE.md` ADR-15), a **manual-annotation sidecar path** for illegible footage (ADR-19),
and **team-colour-aware pass / turnover / assist + a human-marked goal region** (ADR-20). Everything composes
into **one auto command** (`CLAUDE.md` §14, Golden Rule 9). See `CLAUDE.md` §3 and §6.

## Quickstart

```bash
export PATH="$HOME/.local/bin:$PATH"   # uv
make setup                              # create the venv, install core + dev deps
cp .env.example .env                    # fill in ROBOFLOW_API_KEY, optional GEMINI_API_KEY, etc.

# --- Provide input ONE of these ways, then run the SAME command ---

# (a) filename carries the jersey number:  clip2 77.mp4  ->  target = #77
# (b) filename-less, but you know the number:  --target-jersey 22
# (c) filename-less + clear footage:  the pipeline reads the number itself (OCR/VLM)
# (d) poor footage:  drop a sidecar next to the video (see "Manual annotations" below)

make run                                # AUTO FLOW: processes EVERY video in input/, auto-branching per video
make run VIDEO=input/"clip2 77.mp4"     # or process a single video
make run VIDEO=input/match.mp4 TARGET=22  # single video + explicit jersey number

make test                               # unit tests (incl. annotation parser + colour pass/turnover)
make eval                               # detection mAP + tracking HOTA on data/eval/
```

GPU extras (RF-DETR/torch, team-embedding stack) install separately via `make install-gpu` once you're ready to
run real inference — they are not part of the base `make setup`.

## The one auto flow (`CLAUDE.md` §14)

`make run` looks at each video in `input/` and picks a branch with **no human step**:

1. **Is there a `<video>.annotations.txt` sidecar?** → **manual mode**: those annotations are the authoritative
   events; identity (jersey # + colour) comes straight from them; detection/tracking still run so the annotated
   video shows boxes + ball. *The client's file winning is the trigger — the client stays in control of which
   clips are manual.*
2. **Otherwise → auto mode**: jersey number from the filename, or else the ADR-15 extended pipeline always
   runs (optionally given a human-confirmed `--target-jersey`) — its own OCR/VLM verification step *is* the
   honest quality check: it verifies numbers on clear footage and honestly reports which takes it couldn't
   verify on poor footage (naming them in `identity_report.json`), rather than guessing either way.

Either way, events use the **team-colour rules** below and goals use whichever honest source exists.

## Manual annotations (poor-quality footage)

Drop a UTF-8 text file named exactly like the video plus `.annotations.txt` next to it, e.g.
`input/match.annotations.txt`, in the client's own phrasing:

```
# a source URL or comment line is allowed and ignored
https://www.youtube.com/watch?v=wzvM66c8ZmE
Min 26:56 player #2 in white takes a touch
Min 26:58 player #2 in white makes a pass
Min 27:28 player #2 in white kicks the ball out of bounds
```

Format per line: `[Min ]MM:SS player #<number> in <colour> <action>`. Actions map to event types via
`configs/annotations.yaml` (touch, pass, shot, goal, assist, tackle, save, sprint, dribble, out of bounds,
celebration; anything unrecognized is kept as an "other key moment" and reported, never dropped). See
`CLAUDE.md` §14.3. `scripts/make_annotation_template.py` can emit a blank template pre-filled with the takes a
prior auto run couldn't verify.

## Event logic (team-colour aware — `CLAUDE.md` ADR-20)

- **Pass** — the target plays the ball to a receiver wearing the **same jersey colour** (same team cluster) →
  counted as a completed pass.
- **Turnover** — the ball goes to the **other** colour → logged as a turnover, **not** counted as a pass
  (hiding it would be a silent, dishonest filter).
- **Assist** — a pass to a teammate who then **scores** within a short window → credited to the passer.
- **Goal** — sourced honestly from any of: a **manual annotation**, a **scoreboard** digit change (broadcast),
  or the ball entering a **human-marked goal-mouth region** for a static camera (`configs/goal_region.yaml`).
  There is **no** trained net detector and **no** "ball vanished in a corner" guess. When none of the three
  sources exists for a clip, goals read **not available** — never a fabricated one.

## I/O contract (`CLAUDE.md` §8, §10, §13.5)

- `input/` — source video(s), auto-detected. Optional sidecars live here too:
  `input/<video>.annotations.txt` (manual events) and the per-video entries in `configs/goal_region.yaml`.
- `work/` — cached, resumable per-stage artifacts (parquet/JSON + overlay video), chunked.
- `output/<slug>/` — the annotated video, plus `players/player_<N>/` folders (stat card, per-category highlight
  reels, machine-readable timeline), `identity_report.json`, and — in manual mode — `annotation_report.json`.
- `configs/*.yaml` — one file per pipeline stage; **no magic numbers in code**, every tunable lives here —
  including the annotation phrase→event map and the colour/goal-region thresholds.
- `data/eval/` — labeled slices for the evaluation harness (`make eval`).

## Data contracts

All stages compose through the pydantic v2 models in `src/common/types.py` (`Frame`, `Detection`,
`BallDetection`, `PitchHomography`, `Take`, `Track`, `Event`, `Clip`, `PlayerStats`, `Annotation`,
`TakeIdentityResult`, `RunReport`, …). No stage invents a field it can't justify — unknowns are explicit
(`None` + a confidence), never guessed. See `CLAUDE.md` §4.

## High-quality control input (SoccerNet)

`scripts/download_soccernet.py` is kept as the project's **good-quality** control input — "if we need
high-quality video we can take it." `--list` needs no password; the actual download needs `SOCCERNET_PASSWORD`
in `.env` (from the NDA form at soccer-net.org). It's how we prove the auto jersey-read path actually reads
numbers when the footage supports it, cleanly separated from the amateur-footage legibility ceiling. See
`CLAUDE.md` §3.4. The pipeline itself does **not** download YouTube footage — the owner supplies that file plus
its annotations sidecar.

## License hygiene

This project ships **no AGPL dependencies**. In particular: **RF-DETR, not Ultralytics YOLO; `supervision`'s
ByteTrack, not BoxMOT** — both YOLO and BoxMOT are AGPL-3.0 and are never added to any dependency list here
(Golden Rule 6). The manual-annotation parser adds **no new dependency** (stdlib `re` + PyYAML). See
`CLAUDE.md` §7 for the full license table, checked before every new dependency is added.

## How this repo is built (for contributors)

The owner runs Claude Code in **`opusplan`**: **Opus plans in Plan Mode, Sonnet executes.** Plans are thorough;
execution is incremental and resumable. `.claude/commands/process-match.md` runs the flow. See the delivered
Claude Code prompt and `CLAUDE.md` §10.

## Source of truth

**`CLAUDE.md`** — architecture, phases, golden rules, hardware budget, and the decision log (ADRs). Update it in
the same commit as any decision it documents.
