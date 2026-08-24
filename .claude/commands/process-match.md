---
description: Plan and run the Phase-1 pipeline (ingest -> profile -> shots -> detect -> pitch -> track -> team -> events -> manual pick -> highlights -> stats) on video(s) in input/
---

# /process-match

Encodes the CLAUDE.md §5/§6 Phase-1 flow end to end, from raw video in `input/` to a reel + stat
card + run report in `output/`. **Meant to be planned in Plan Mode (Opus) and then executed
(Sonnet)** — CLAUDE.md §10: "the user runs Claude Code in `opusplan` — Opus plans, Sonnet
executes." Read `CLAUDE.md` fully before running this; it is the source of truth and overrides
anything below if they conflict.

## 0. Detect input + parse target jersey

- List `input/*.{mp4,mkv,mov}`. For each file, match it against
  `configs/run.yaml: filename_convention_regex` (`^clip(?P<n>\d+)\s+(?P<jersey>\d+)$` on the
  stem). A match sets `target_jersey` (run metadata only in Phase 1 — Golden Rule 7, CLAUDE.md
  §3.1); a non-match still runs, just without a pre-filled jersey label.
- If `input/` is empty or missing, stop and tell the user what to drop in and where.

## 1. Verify environment + secrets — ASK if missing

- Confirm `.venv` exists (`make setup` if not) and `.env` exists (copy from `.env.example` if
  not, then **stop and ask** the user to fill it in — never invent/mock a key).
- Phase 1 needs `ROBOFLOW_API_KEY` at minimum (per ADR-2's checkpoint-resolution ladder), and
  `HF_TOKEN` only if the chosen model card turns out to be gated. If either required key is
  blank when a stage needs it, **stop and ask**: what the key is for, why it's needed now, and
  where to get it. Never skip a stage silently (CLAUDE.md §10).
- Run the VRAM preflight (`configs/hardware.yaml: preflight_warn`) — warn, don't fail, if free
  VRAM is below the stage budget (CLAUDE.md §11.1 notes an unrelated process can hold ~6 GB).

## 2. Run stages 0 -> 6, resumable, per video

Every stage caches its artifact under `work/<clip>_<jersey>/` (via `src.common.io.StageCache`)
and is skipped on a cache hit (matching config hash) — safe to re-run after a crash.

| # | Stage | Invocation (once wired) | Config |
|---|---|---|---|
| 0 | Ingest | `$(PYTHON) -m src.ingest.run --video "<path>"` | `configs/ingest.yaml`, `configs/hardware.yaml` |
| 0.5 | Source profiler | `$(PYTHON) -m src.pipeline.profile_cli "<path>"` (also `make profile`) | `configs/profile.yaml` |
| 1 | Shots (broadcast only; auto-skipped for single-camera) | `$(PYTHON) -m src.shots.run --video "<path>"` | `configs/shots.yaml` |
| 2 | Detect (players/ball, pretrained RF-DETR + SAHI, **no training**) | `$(PYTHON) -m src.detect.run --video "<path>"` | `configs/detect.yaml` |
| 2 | Pitch homography (assisted manual calibration for single-camera, ADR-3) | `$(PYTHON) -m src.pitch.run --video "<path>"` | `configs/pitch.yaml` |
| 3 | Track + team (within-take ByteTrack + SigLIP/UMAP/KMeans) | `$(PYTHON) -m src.track.run --video "<path>"` | `configs/track.yaml`, `configs/team.yaml` |
| 5 | Events without identity (goals/shots/sprints) | `$(PYTHON) -m src.events.run --video "<path>"` | `configs/events.yaml` |
| — | **User selects the track** for the labeled jersey (human-in-the-loop, Golden Rule 4) | interactive prompt, labeled "pick the track for #`<jersey>`" | — |
| 6 | Highlights: clip cut + dedupe + best-first reel export | `$(PYTHON) -m src.highlights.run --video "<path>" --track-id <id>` | `configs/highlights.yaml` |
| 6 | Stats: simple stat card with confidences | `$(PYTHON) -m src.stats.run --video "<path>" --track-id <id>` | `configs/events.yaml` |
| — | Eval: mAP + HOTA on labeled slice (if `data/eval/` has data for this clip) | `make eval` | `configs/eval.yaml` |

After each stage: emit its smoke-test overlay video (`src/common/viz.py`) and log everything
dropped (`src/common/logging.py::DropCounter`) — CLAUDE.md §10. Log a `RunReport` (numbers +
`config_hash`) at the end.

## 3. Write outputs

`output/<clip>_<jersey>/`:
- the exported reel (deduplicated, best-first per `configs/highlights.yaml` rank weights)
- the stat card (`PlayerStats`, every count carries a `confidences[...]` entry — never a bare
  number, Golden Rule 5)
- the run report (`RunReport`: timings, dropped counts, config hash, profile)

## 4. Report back

Summarize: which stages ran vs. were skipped (cache hits, or broadcast-only stages skipped for
single-camera input), what got dropped and why, the manual track pick, and where the outputs
landed. Never claim automatic player identification in Phase 1 (CLAUDE.md §12).
