---
description: Plan and run the Phase-1 pipeline (ingest -> profile -> shots -> detect -> pitch -> track -> team -> events -> manual pick -> highlights -> stats) on video(s) in input/
---

# /process-match

Encodes the CLAUDE.md §5/§6 Phase-1 flow end to end, from raw video in `input/` to a reel + stat
card + run report in `output/`. **Meant to be planned in Plan Mode (Opus) and then executed
(Sonnet)** — CLAUDE.md §10: "the user runs Claude Code in `opusplan` — Opus plans, Sonnet
executes." Read `CLAUDE.md` fully before running this; it is the source of truth and overrides
anything below if they conflict.

## 0. Detect input + branch (CLAUDE.md §14.1 — check in this exact order, per video)

- List `input/*.{mp4,mkv,mov}`. If `input/` is empty or missing, stop and tell the user what to
  drop in and where. For each file (`make run` does this automatically; `make run VIDEO="<path>"`
  targets one):
  1. **Sidecar present?** `input/<basename>.annotations.txt` next to the video ⇒ **ADR-19 manual-
     events mode** — the parsed sidecar is the sole, authoritative event source (no auto
     detectors run for event generation); identity (jersey + colour) comes straight from it.
     This is checked FIRST and wins unconditionally — the client's own file is the trigger, never
     a quality score. `scripts/make_annotation_template.py <video>` can generate a starter
     sidecar from a prior run's `identity_report.json`.
  2. **Else, does the filename match `configs/run.yaml: filename_convention_regex`**
     (`^clip(?P<n>\d+)\s+(?P<jersey>\d+)$` on the stem)? ⇒ the original Phase-1 flow below,
     `target_jersey` set from the filename (Golden Rule 7: run metadata only, human still picks
     the track).
  3. **Else** ⇒ ADR-15's extended pipeline runs regardless of quality (its own OCR/VLM
     verification step IS the honest check — it verifies on clear footage, honestly reports
     unverified takes on poor footage). Pass a human-confirmed number with
     `make run VIDEO="<path>" TARGET=<jersey>` (ADR-18's `--target-jersey`) if you already know it.

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
| 1 | Shots: cut/take segmentation (**every profile**, ADR-7); replay/close-up/scoreboard-OCR (broadcast only) | `$(PYTHON) -m src.shots.run --video "<path>"` | `configs/shots.yaml` |
| 2 | Detect (players/ball, pretrained RF-DETR + SAHI, **no training** — soccer checkpoint per ADR-8, dev/eval only per ADR-9) | `$(PYTHON) -m src.detect.run --video "<path>"` | `configs/detect.yaml` |
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

Summarize: which stages ran vs. were skipped (cache hits, or broadcast-only sub-stages skipped
for single-camera input — cut detection itself still runs per ADR-7), what got dropped and why,
the manual track pick, and where the outputs landed. Never claim automatic player identification
in Phase 1 (CLAUDE.md §12).

## Notes on the current footage (CLAUDE.md §3.2 — check before assuming the general case)

The 5 clips in `input/` are youth/amateur single-camera Veo exports, not broadcast, and several
ADRs exist specifically because of what was measured in them:
- **ADR-7**: `clip4` is single-camera but contains a real cut at t=60.07s (two different
  matches) — cut detection must never be skipped just because a clip "looks" single-take.
- **ADR-6**: this footage often has fewer than 4 usable pitch landmarks; sprint speed must be
  reported as uncalibrated (pixel units, confidence-penalised) rather than fabricated metres.
- **§3.2(3)**: there is no scoreboard anywhere in this footage, so the goal detector must report
  "not available" here, never a guessed goal.
- **§3.2(1)**: `clip2`/`clip4` carry a burned-in red-arrow graphic pointing at the target player
  early on — a detection hazard *and* a free (not-yet-implemented) manual-selection aid.
Re-check CLAUDE.md §3.2 directly if any of this seems to have changed — it is updated live as the
footage gets inspected further.
