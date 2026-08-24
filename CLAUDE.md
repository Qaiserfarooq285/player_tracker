# CLAUDE.md — AI Soccer Highlight Analyzer

> Source of truth for how we build this. Keep it current: when a decision changes, update this file in the
> same commit. Read it fully before working.

## 1. What this project is
Ingest a **full broadcast soccer match** (90+ min, TV-style: multiple cameras, cuts, zooms, replays,
on-screen graphics) and produce **per-player highlight reels** + a **basic stat card**.
User flow: `Upload match → choose a player → process → review actions → auto/select highlights → export reel`.

Input reality we design for: **broadcast** (hardest case) and a **production** quality bar.
Reference blueprint (rationale for every choice): https://claude.ai/code/artifact/e3f76bf9-ac19-4d68-8175-f4bfba418f37

## 2. Golden rules (do not break without asking the owner)
1. **Pipeline of specialized models, not one model.** Build stage by stage (§5).
2. **Never train from scratch.** Use pretrained models and **fine-tune** on public soccer data. If from-scratch
   training seems necessary, stop and ask.
3. **Broadcast tax first.** Segment video into continuous **camera takes** and drop replays/close-ups before
   any tracking. Track only *within* a take; reset IDs at every cut.
4. **Human-in-the-loop for player identity.** Cross-cut auto-ID is not production-reliable — design a
   user-confirmation step. Never claim perfect automatic identification.
5. **Every stat is traceable + carries a confidence.** No fabricated numbers; each event links to its clip.
   Lead with reliable events (goals, shots, sprints); passes/touches/tackles are best-effort, flagged.
6. **License hygiene.** Prefer Apache-2.0 / MIT / BSD. **RF-DETR, not Ultralytics YOLO. `supervision` tracker,
   not BoxMOT.** Flag any GPL/AGPL/research-only dependency in §7 before adding it.
7. **Ship Phase 1 first** (§6). No jersey auto-ID or fine-grained action stats until Phase 1 is done + evaluated.
8. **Evaluation is part of the build.** Stand up the eval harness early; verify with numbers, not vibes.

## 3. Current phase
**▶ PHASE 1 — within-take core with MANUAL player selection.** (See §6 for scope + definition of done.)
Update this line as we advance.

### 3.1 Current input set (measured 2026-08-24)
The owner's footage is **not** a broadcast match — it is 5 short **4K single-camera** clips. This is exactly the
"validate on a single-camera clip first" case in §6. The source profiler (Stage 0.5) is still wired from day one
so the broadcast branch stays automatic.

| File | Resolution | FPS | Duration | Frames | Target jersey |
|---|---|---|---|---|---|
| `input/clip1 43.mp4` | 3840×2160 | 30 | 11.1 s | 372 | **43** |
| `input/clip2 77.mp4` | 3840×2160 | 30 | 13.4 s | 415 | **77** |
| `input/clip3 77.mp4` | 3840×2160 | 30 | 13.5 s | 414 | **77** |
| `input/clip4 77.mp4` | 3840×2160 | 30 | 110.9 s | 3448 | **77** |
| `input/clip5 77.mp4` | 3840×2160 | 30 | 14.3 s | 471 | **77** |

**Filename convention:** `clip<N> <jersey>.mp4`. The trailing integer is the **target player's jersey number**.

**How the jersey number is used in Phase 1 (important):** it is **run metadata only**. Per Golden Rule 7 we do
**not** run jersey OCR / auto-ID in Phase 1. The number is parsed from the filename into the run config
(`target_jersey`), used to name outputs (`output/clip1_43/…`) and to label the manual-selection prompt
("pick the track for #43"). The human still selects the track. Auto-ID by number lands in **Phase 2** (§6),
and the seam (`src/identity/`, `Track.jersey_number`, `Track.id_confidence`) exists from day one.

**4K note:** frames are decoded at 3840×2160 and **downscaled per stage** (see `configs/hardware.yaml`).
Never feed 4K straight to a detector — it wastes VRAM for no accuracy gain at these player sizes. Full-res
crops are used only where small-object detail matters (ball via SAHI, and Phase-2 jersey OCR).

### 3.2 Footage characteristics (inspected 2026-08-24 — these drive real design changes)
The clips are **Veo** exports (Veo Technologies AI camera; "veo" watermark bottom-right). Youth/amateur matches,
elevated wide sideline camera, digital pan/zoom cropped out of a stitched panoramic.

**Measured** *(corrected 2026-08-24 after Stage 0.5 shipped — see the two ⚠️ corrections below)*:
- **Cuts** — clip1/2/3/5 → **0 cuts** (single take each). **clip4 → 3 cuts at 60.13 s / 73.63 s / 87.30 s
  ⇒ 4 takes.** clip4 is a **4-venue compilation** of player #77's highlights across different matches
  (grass → turf-with-running-track → cold-weather grass with cones → bright grass).
  > ⚠️ **Correction.** An earlier ffmpeg `scdet @0.35` pass recorded here found only **1** cut and concluded
  > "2 takes". That threshold was too strict and **missed two real cuts**. PySceneDetect found them;
  > frame contact-sheets at 72.5/74.5/86.0/88.5 s confirm four visually distinct venues, and a re-run of
  > `scdet` at 0.15 reproduces all three. **Lesson: a single detector at one threshold is not ground truth.**
  > `configs/shots.yaml` threshold is now **38.0**, set from the *measured* content-value gap on this data
  > (real cuts ≥ 45.6, false positives ≤ 32.0) — not tuned to match a prediction. The rejected spikes at
  > 14.4 / 47.0 / 99.6 s are fast pans and foreground crossings, verified same-venue either side.
- **Camera motion**: sparse KLT + `estimateAffinePartial2D` (RANSAC) on consecutive native frames,
  median-aggregated. Scores (px/frame): clip1 **12.70**, clip2 **3.79**, clip3 **1.95**, clip4 **3.14**,
  clip5 **3.12**.
  > ⚠️ **Correction.** An earlier `vidstabdetect` eyeball recorded "≈ ±1 px/frame, near-static". That was
  > read off raw local-motion vectors and was **wrong**. Dense Farneback was also tried and rejected: its
  > score proved wildly resolution-dependent (0.07 at 4K vs 6.1 at 240 px on the same clip) — textureless
  > grass gives it nothing to lock onto.
- **⚠️ Every clip opens with a genuine ~3–5 s camera-settling pan** before going steady. Undocumented until
  Stage 0.5 measured it. Median aggregation absorbs it on long clips, but on an 11 s clip it is ~36% of the
  runtime, which is why **clip1 reads `single-panning`**.
- **Player scale** varies hugely within a frame: near players ≈ 450 px tall at 4K, far players ≈ 60 px.
- **Jersey numbers are legible at 4K** for mid/near players (#77, #22, #34, #13, #8 all readable even at 1280 px).

**Consequences (each one changes a stage):**
1. **⚠️ A burned-in red arrow graphic** marks the target player in the opening seconds of clip2 and clip4 (gone
   by t=60 in clip4). It is an editorial overlay, not an object. It will (a) generate spurious detections and
   (b) partially occlude the target. Stage 2 must mask/ignore it. It is *also* a free prior for locating the
   target player — a legitimate Phase-1 aid for **manual** selection (it does not read a jersey number, so it
   does not violate Golden Rule 7). Tracked as a task, not yet implemented.
2. **Takes ≠ profile.** clip4 proves a single-camera source can still contain cuts. **Revision to §5:**
   shot-boundary detection now runs **always** (it is cheap and clip4 needs it); only the *replay detector,
   close-up classifier, and scoreboard OCR* are gated on `profile == broadcast`. The original "skip Stage 1
   entirely for single-camera" rule would have silently corrupted clip4's tracking. IDs still reset at cuts.
3. **⚠️ No scoreboard anywhere in this footage** → the Phase-1 "goal = scoreboard delta" detector is **N/A**
   here. It stays in the code for broadcast input, but on these clips it must report *"not available"*, never a
   guessed goal (Golden Rule 5). Usable Phase-1 events on this footage: **sprints** and **shots (heuristic)**.
4. **⚠️ Sparse/ambiguous pitch lines.** clip1 is a shared-use field with **American-football lines painted over
   the soccer pitch** (yard numbers, hashes); clip2/4 show only a touchline and part of a box, with portable
   goals. A pitch-keypoint model trained on pro stadiums will fail on both. This strongly confirms **ADR-3**.
   But note a static camera is *not* enough on its own — where fewer than 4 usable landmarks are visible,
   **do not fabricate metres**: emit speed with `calibrated: false`, a confidence penalty, and pixel-normalised
   units. See ADR-6.
5. **Detection input size**: at 1280 px the far players are ~30 px tall — marginal. Plan to run detection at
   **1536–1920 px** and/or apply **SAHI tiling to players as well as the ball**, then measure. The
   `image_size` knob in `configs/hardware.yaml` is the place to tune this.

## 4. Data contracts (`src/common/types.py`, pydantic)
Stages compose through these; intermediate artifacts cache to disk (parquet/JSON + video) so stages run independently.
- `Frame(index, t, path)`
- `Detection(bbox, cls, conf)` · `BallDetection(bbox, conf, interpolated: bool)`
- `PitchHomography(matrix, keypoints, conf)` — pixels ↔ pitch metres
- `Track(id, take_id, boxes[], team?, jersey_number?, id_confidence, path_pitch_xy[])`
- `Event(type, t_start, t_end, player_track_id?, confidence, source)` — `source` = how it was derived
- `Clip(event_id, t_start, t_end, take_id, rank_score, path)`
- `PlayerStats(player_ref, counts{...}, confidences{...})`
Rule: no stage invents fields it can't justify; unknowns are explicit (`None` + confidence), never guessed.

## 5. Architecture (build order; models are current recommendations — verify latest before committing)
| Stage | Job | Use | Notes / license |
|---|---|---|---|
| 0 Ingest | decode, sample fps | `ffmpeg` (NVDEC), PyAV | per-stage sampling; 90min@25fps ≈ 135k frames |
| 0.5 Source profiler | detect input type | cuts (PySceneDetect) + camera-motion + resolution → run profile | `broadcast` / `single-static` / `single-panning`; broadcast path is a **superset** (single-cam = one long take) |
| 1 Temporal structure | takes, replays, shot type, clock | **PySceneDetect** (→ TransNetV2), replay + shot-type classifier, **PaddleOCR/EasyOCR** | **only when profile=`broadcast`; auto-skipped for single-camera**; protects everything downstream |
| 2 Perception | players/ball/refs, pitch | **RF-DETR** (Apache), **SAHI** tiling / **TrackNetV3** ball, pitch keypoints + `supervision` ViewTransformer | ball is the weak spot; homography enables speed |
| 3 Track + team | IDs within take, team | **`supervision` ByteTrack (MIT)**, associate in pitch coords; **SigLIP→UMAP→KMeans** team | reset IDs at cuts; same-kit ⇒ weak appearance |
| 4 Identity *(Phase 2)* | who is #N | **ViTPose**→legibility→**PARSeq**→tracklet vote (ref `mkoshkina/jersey-number-pipeline`); VLM fallback; fuse #+team+ReID+position → `id_confidence` → **HITL** | ~87% per-tracklet ceiling; route low conf to human |
| 5 Events | actions | P1: goals (scoreboard Δ), shots (heuristic), sprints (homography+speed). P2+: **T-DEED** action spotting; assist = rule (inferred); optional Gemini/TwelveLabs | ~60 mAP@1 SOTA for ball actions |
| 6 Assemble | rank, cut, reel, stats | rules score + optional VLM; `ffmpeg` cut (snap to takes, dedupe); stat card w/ confidence + click-to-clip | user picks clip count/length |

## 5.1 Decision log (ADRs — deviations from the original brief, with reasons)
Anything here overrides the brief's default suggestion. Add a row whenever a recommendation is changed.

| # | Decision | Why | Status |
|---|---|---|---|
| ADR-1 | **PyAV + ffmpeg CLI for decode; drop `decord`** | `decord` 0.6.0 has no cp311/cp312 wheels and is effectively unmaintained; building it from source is a needless risk. PyAV covers seeking/frame-accurate decode, and `ffmpeg -hwaccel cuda -c:v h264_cuvid` covers NVDEC bulk decode. Same capability, zero build risk. | ✅ adopted |
| ADR-2 | **Roboflow's soccer player/pitch checkpoints are YOLOv8 → AGPL → cannot ship** | `roboflow/sports` is MIT *code*, but the released player-detection and field-keypoint **weights are Ultralytics YOLOv8**, which is AGPL-3.0. Using them violates Golden Rule 6. Resolution ladder: (a) an **Apache-2.0 RF-DETR** soccer checkpoint; (b) **RF-DETR COCO-pretrained** (`person` + `sports ball`); (c) optional overnight fine-tune. **No training in Phase 1 either way.** | ✅ **resolved → see ADR-8** |
| ADR-8 | **Phase-1 detector = `julianzu9612/RFDETR-Soccernet` (Apache-2.0), with RF-DETR-COCO as fallback** | Investigated 2026-08-24 with the validated Roboflow key. **(1)** The three canonical Roboflow soccer projects (`football-players-detection-3zvbc`, `football-field-detection-f07vi`, `football-ball-detection-rejhg`) return `model: None` on **every** version — Roboflow hosts **no trained weights** for them, only data. Their *datasets* are **CC BY 4.0** (commercially usable with attribution) — so they are excellent **eval + future fine-tune** material, which is how we use them. **(2)** On HF, `julianzu9612/RFDETR-Soccernet` is **Apache-2.0**, RF-DETR-Large (128 M, DINOv2 backbone, 1280², 1.46 GB) with exactly the classes we need — `ball, player, referee, goalkeeper` — reporting mAP@50 **0.857** / mAP **0.498** on SoccerNet. **(3)** The alternative `OrbitalLab/mova-rfdetr-soccernet-v1` (MIT) is **gated** (needs an access request) → skipped. | ✅ adopted for P1 |
| ADR-9 | **⚠️ SoccerNet-derived weights are dev-only until licensing is cleared** | ADR-8's checkpoint is *declared* Apache-2.0 by its uploader, but it was **trained on SoccerNet-Tracking-2023**, and §7 lists SoccerNet data as **research/education only**. Whether an uploader can relicense a model trained on restricted data is legally unsettled — so this is a Golden-Rule-6 "flag before adding", not a silent adoption. **Use it for Phase-1 development and evaluation** (internal R&D, not distribution). **Before any commercial ship**, take one of: (i) obtain SoccerNet commercial terms, (ii) fine-tune RF-DETR on the **CC BY 4.0** Roboflow soccer dataset, or (iii) fall back to RF-DETR-COCO. Do not ship (i)-unresolved. | ⚠️ open — revisit before ship |
| ADR-11 | **Downstream stages must consume the *motion score*, not the static/panning *label*** | Stage 0.5 put **4 of 5 clips inside the ambiguous band** (1.5–6.0 px/frame). clip2 scores **3.79 → `single-panning`** while clip4 scores **3.14 → `single-static`**, purely because the band's midpoint is 3.75 — essentially identical footage landing on opposite sides of a knife-edge. Worse, the label is contaminated by the ~3–5 s settling pan every clip opens with (§3.2), which is a startup transient, not a camera regime. Acting on the label would mean "calibrate homography once" vs "re-estimate continuously" flipping on noise. **Resolution:** the label stays in `RunProfile` for reporting, but **ADR-3's homography cadence reads `motion_score` directly** — recalibration interval scales continuously with measured motion, with the settling window excluded. For Veo footage the honest answer is that the camera digitally pans to follow play, so homography needs periodic re-estimation *regardless* of which label it got. | ✅ adopted — implement at Stage 2/pitch |
| ADR-10 | **Expect a real domain gap; do not trust the published 0.857 mAP on this footage** | ADR-8's checkpoint was trained on SoccerNet = **professional broadcast** footage. §3.2 footage is **youth/amateur Veo** — higher/wider fixed camera, smaller players, different kits, painted-over American-football lines. Published mAP does **not** transfer. This is precisely why §9's eval harness is built before tuning: measure mAP on *our* labeled slice, and treat the Roboflow CC BY 4.0 set as a second eval set. | ✅ adopted |
| ADR-3 | **Phase-1 homography = assisted 4-point manual calibration for single-camera profiles** | The usual auto pitch-keypoint model is also YOLOv8-pose (AGPL, same problem as ADR-2). For a *static* camera one calibration serves the whole clip, it is more accurate than a per-frame keypoint model, it costs no VRAM, and it is consistent with the human-in-the-loop principle (Golden Rule 4). Auto keypoints (PnLCalib / TVCalib / an Apache RF-DETR-pose) get evaluated for the broadcast branch. | ✅ adopted for P1 |
| ADR-4 | **Repo root = the existing working directory**, not a new `soccer-highlight-analyzer/` subdir | `input/` already exists here and holds the owner's clips; nesting would orphan them. | ✅ adopted |
| ADR-5 | **Python 3.11 provisioned via `uv`** (system has only 3.12) | Honors §10 without `sudo` or deadsnakes; `uv` also gives fast, reproducible locking. | ✅ adopted |
| ADR-6 | **Uncalibrated speed is reported as uncalibrated, never converted to fake m/s** | §3.2(4): on this footage there are often <4 usable pitch landmarks, so no metric homography exists. Golden Rule 5 forbids fabricated numbers. `Event.evidence` carries `calibrated: bool`; when false, speed is in normalised-pixel units, the confidence is penalised, and the UI/stat card must say "uncalibrated". A sprint can still be *ranked* without metres. | ✅ adopted |
| ADR-7 | **Shot-boundary detection runs for every profile, not just `broadcast`** | §3.2(2): clip4 is single-camera yet contains a real cut at 60.07 s. Gating cut detection on `profile == broadcast` would have let tracking run straight across it and corrupt IDs. Only replay/close-up classification and scoreboard OCR stay broadcast-gated. | ✅ adopted, supersedes §5 row 1 |

## 6. Phases
**Phase 0 (done as part of P1 kickoff):** env check, repo scaffold, data contracts, eval harness stub, license table.

**Phase 1 — within-take core, MANUAL player pick (current):**
Ingest → shot-boundary + drop replays → detection (**pretrained** RF-DETR/soccer weights + SAHI ball; **no
training**) → pitch homography → within-take tracking + team → events without identity (goals/shots/sprints) →
**user selects a track** → clip cut + dedupe → best-first reel export → simple stat card → eval (mAP + HOTA on
labeled slice) → thin API + minimal UI. **Runs entirely on the local A2000 12 GB — inference only, no paid cloud.**
Handle **any single uploaded video**: the source profiler auto-detects broadcast vs single-camera and branches
(single-camera skips Stage 1, calibrates homography once, keeps IDs across the whole clip — the easy case).
**Validate on a static single-camera clip first**, then broaden to broadcast.
*Definition of done:* on a sample match + a manually-selected player, output a de-duplicated reel + stat card
(with confidences) + an eval report, reproducible via `make run`.

**Phase 2 — player identity:** jersey OCR + cross-take re-association + human-in-the-loop confirm → follow a
chosen number automatically; per-player event attribution; passes/touches as best-effort (flagged).
*This is where `target_jersey` (§3.1) stops being a label and starts driving automatic selection.*

**Phase 3 — hardening:** fine-tuned action spotting; tackles/saves/assists; confidence + correction UI; scale
compute; QA vs ground truth; optional provider-data adapter; active-learning loop (fixed errors → retrain).

## 7. Dependency license table (keep updated; ⚠️ = do not ship in closed product without action)
| Component | License | OK for closed product? |
|---|---|---|
| RF-DETR (`rfdetr`) | Apache-2.0 | ✅ |
| `supervision` (tracking/annot.) | MIT | ✅ |
| SAHI | MIT | ✅ |
| PySceneDetect | BSD-3 | ✅ |
| PyAV | BSD-3 | ✅ |
| MMPose / MMOCR (ViTPose) | Apache-2.0 | ✅ |
| PARSeq | Apache-2.0 | ✅ |
| PaddleOCR / EasyOCR | Apache-2.0 | ✅ |
| SigLIP weights (via `transformers`/`timm`) | Apache-2.0 | ✅ (verify weight card) |
| TransNetV2 | MIT | ✅ |
| TrackEval | MIT | ✅ |
| Ultralytics YOLO | **AGPL-3.0** | ⚠️ avoid — use RF-DETR |
| **Roboflow soccer player/field *weights* (YOLOv8)** | **AGPL-3.0** | ⚠️ **do not ship** — see ADR-2 |
| `roboflow/sports` (code only) | MIT | ✅ code ok; weights are the problem |
| **`julianzu9612/RFDETR-Soccernet` weights** | Apache-2.0 *(declared)* | ⚠️ **dev/eval only** — trained on SoccerNet; see **ADR-9** before shipping |
| Roboflow `football-*-detection` **datasets** | **CC BY 4.0** | ✅ commercial OK **with attribution** — our eval + fine-tune source |
| `OrbitalLab/mova-rfdetr-soccernet-v1` | MIT | ⛔ gated on HF (access request) — not used |
| BoxMOT | **AGPL-3.0** | ⚠️ avoid — use `supervision` tracker |
| SoccerNet data | **Research/education** | ⚠️ email for commercial terms |
| Broadcast footage | Rights-holder owned | ⚠️ legal review before commercial use |
Audit each new dependency's license here **before** adding it.

## 8. Commands (implement these; keep accurate)
```
make setup     # create env, install deps, download model weights
make run       # process the video in input/ → reel + stat card + report in output/ (resumable via work/)
make eval      # detection mAP + tracking HOTA (+ action mAP@1 later) on data/eval/
make test      # unit tests (contracts, ranking, dedupe, speed calc)
make lint      # ruff + black
scripts/download_soccernet.py   # pulls SoccerNet subsets (needs NDA password in .env)
scripts/pull_roboflow.py        # pulls datasets + pretrained soccer weights (needs ROBOFLOW_API_KEY)
scripts/prepare_eval.py         # ingest labeled slices into data/eval/
```
Secrets in `.env` (`HF_TOKEN`, `ROBOFLOW_API_KEY`, `SOCCERNET_PASSWORD`, optional `GEMINI_API_KEY`,
`TWELVELABS_API_KEY`); never commit them.

## 9. Evaluation (metrics + first targets — refine against your labeled set)
- **Detection:** mAP@50 on your labeled slice (players first, ball tracked separately — expect ball lower).
- **Tracking:** **HOTA** via **TrackEval** (report IDF1 + ID switches too), measured **within takes**.
- **Actions (Phase 2+):** **mAP@1s** (SoccerNet ball-action style).
- **Jersey (Phase 2):** tracklet accuracy.
Wire these before optimizing anything. Log a run report (numbers + config hash) per run.

### 9.1 Eval datasets on disk
| Set | Path | Content | Licence | What it measures |
|---|---|---|---|---|
| **Roboflow football-players v20** | `data/eval/roboflow-football-players-v20/` | 372 imgs / 8,905 anns (train 298 · valid 49 · test 25); classes `ball, goalkeeper, player, referee` | **CC BY 4.0** — attribution required, see `ATTRIBUTION.md` | Detection mAP **baseline** |
| **Our Veo slice** | `data/eval/veo-labeled/` | ⛔ **not yet created** — must be hand-labeled from `input/` | own footage | Detection mAP + HOTA **that actually counts** |

> ⚠️ **The Roboflow set is a sanity baseline, not our number.** Its images are **576×576 crops of
> professional broadcast** footage. Per **ADR-10**, results there do **not** transfer to youth/amateur Veo
> footage with a high fixed camera and much smaller players. A Phase-1 "definition of done" mAP/HOTA claim
> must be measured on a **hand-labeled slice of our own clips** — creating that slice is a Phase-1 task, not
> an optional extra. Report both numbers side by side so the domain gap stays visible.
>
> Note: dataset **v20 is named "rf-detr-m"** — the publisher prepared it for RF-DETR training, which
> corroborates ADR-8's model choice and makes it the natural base for the ADR-9 clean-licence fine-tune.

## 10. Working conventions
- Python 3.11 (via `uv`, see ADR-5). One `configs/*.yaml` per stage; **no magic numbers in code.**
- Cache each stage's output to disk; stages must be independently runnable/debuggable.
- After each stage: a **smoke test** on a short clip + an **annotated overlay video** artifact.
- **Log everything dropped** (replays, close-ups, low-confidence). Silent filtering = hidden bugs.
- Commit per stage; update this file when decisions change; verify with eval numbers before "done".
- **I/O contract:** input video in `input/` (auto-detected); cached artifacts in `work/` (resumable, chunked);
  final reel + stat card + run report in `output/`. Gitignore `input/`, `work/`, `output/`, `.env`, `models/`, `data/`.
- **Secrets: ask, never assume.** Load from `.env`; when a token/decision is missing, STOP and ask the user (what /
  why / where) — never mock data or skip a stage silently. Likely Phase-1 asks: Roboflow key, maybe HF token.
- **Model workflow:** the user runs Claude Code in **`opusplan`** — Opus plans (Plan Mode), Sonnet executes. Plan
  thoroughly; keep execution incremental + resumable. Provide `.claude/commands/process-match.md` to run the flow.

## 11. Hardware profile & compute budget
**Single machine: NVIDIA RTX A2000 12 GB** (Ampere, low-power workstation; has NVENC/NVDEC). **LOCAL ONLY — no
paid cloud.** The **entire pipeline, including full 90-min matches, runs on this GPU; long runtimes are acceptable**
(design for "slow but completes", not real-time). Because **Phase 1 is inference-only with pretrained models, no
training is required** and 12 GB is comfortable. Any fine-tuning is deferred/optional and, if ever needed, runs
overnight on the A2000 or on a **free** GPU tier (Google Colab / Kaggle Notebooks, ~16 GB T4/P100 — check current limits).
Never gate progress on paid GPUs.

### 11.1 Measured environment (2026-08-24)
| Item | Value |
|---|---|
| GPU | NVIDIA RTX A2000 12 GB (12282 MiB), Ampere |
| Driver / CUDA | 580.173.02 / CUDA 13.0 runtime; `nvcc` 12.0 |
| ffmpeg | 6.1.1 — `h264_cuvid` **NVDEC** ✅, `h264_nvenc` **NVENC** ✅, `-hwaccel cuda` ✅ |
| Python | system 3.12.3; **project uses 3.11 via `uv`** (ADR-5) |
| RAM / disk | 125 GB / 124 GB free |

> ⚠️ **VRAM contention:** an unrelated process (`Downloads/Ad Blocker/.venv`, PID 10393) was holding **6.0 GB**
> and 87% GPU util at scaffold time, leaving only ~5.2 GB free. `configs/hardware.yaml` therefore defaults to a
> **5 GB budget**, and the pipeline runs a **VRAM preflight** that warns (not fails) when free VRAM is below the
> stage budget. Free that process before long runs to use the full 12 GB.

Fit-in-12 GB rules (put the knobs in `configs/hardware.yaml`, 12 GB-safe defaults):
- **Run stages sequentially, cache to disk between them** — never hold detector + pose + SigLIP + OCR in VRAM at once; explicitly free/unload each model before loading the next.
- **Chunk the match** (per camera take, or fixed 2–5 min segments) and make every run **resumable** from cached artifacts — a full match runs overnight and survives a crash/restart without redoing finished work.
- **Mixed precision (AMP/FP16)** everywhere; add an **ONNX / TensorRT** inference path (Ampere → big speedup; worth it, we're compute-bound).
- **Small / inference-friendly variants:** RF-DETR-base/small (not large), ViTPose-small, ByteTrack (cheap). Inference **batch size 1–2**.
- **VLM layer via API** (Gemini / TwelveLabs free tiers), not a local VLM — an 11B vision model won't fit in 12 GB.
- **Decode with NVDEC** (`ffmpeg`), **sample frames per stage**, and **skip replays/close-ups** so you process only ~⅓–½ of raw frames — the main lever that makes a full match tractable.
- **Downscale 4K before detection** (§3.1) — 3840×2160 buys nothing at these player sizes and costs ~4× the VRAM.

Per-stage fps sampling defaults (tune on your footage):
shot-boundary/scoreboard **1–2 fps** · detection+tracking **5–12 fps** · ball + action windows **full fps only where needed** · team/jersey **1–2 fps per tracklet**.

Estimate runtime honestly: **time a 2-min clip end-to-end, then multiply (~45× for 90 min).** Print a **per-stage VRAM +
wall-clock report** so bottlenecks are visible. A full match taking a few hours is fine — it's a background batch job.
If a stage OOMs: batch→1, lower image size, or switch to a smaller variant.

## 12. Don'ts
❌ Train from scratch. ❌ Ship YOLO/BoxMOT (AGPL). ❌ Auto-ID player in Phase 1. ❌ Pass/touch/tackle stats in
Phase 1. ❌ Track across cuts. ❌ Emit untraceable stats. ❌ Build everything at once. ❌ Hard-code params.
❌ Over-promise accuracy in UI copy. ❌ Feed raw 4K to a detector.
