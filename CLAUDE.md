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
**▶ PHASE 1 — within-take core with MANUAL player selection**, plus an **owner-authorized, verified-jersey
auto-ID exception for filename-less inputs** (ADR-15) that pulls a narrow slice of Phase 2 forward. (See §6
for scope + definition of done.)
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
1. **⚠️ A burned-in red arrow graphic** marks the target player in the opening seconds of clip2 and clip4.
   It is an editorial overlay, not an object: it will (a) generate spurious detections and (b) partially
   occlude the target. Stage 2 must mask it. It is *also* a free prior for locating the target player — a
   legitimate Phase-1 aid for **manual** selection (it reads no jersey number, so Golden Rule 7 holds).
   **Measured signature (clip2, 3840×2160):** at t=3.0 s a saturated-red HSV mask
   (`H∈[0,10]∪[170,180], S>120, V>90`) yields **exactly one** connected component >2000 px —
   310×606 px at (2020, 262), area 27,756 (0.34% of frame). At t=4.0 s only 317 red px remain: **the arrow
   is transient**, present for roughly the first few seconds only. So the rule is robust: take the single
   largest saturated-red component above `arrow_min_area_px`, mask it, and use its **tip** as the
   target-player prior; when no such component exists, there is simply no arrow that frame. Thresholds go in
   `configs/detect.yaml`, never inline.
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

### 3.3 Second input: `Jordan Thomas Highlight Video.mp4` (added 2026-08-27 — no filename jersey metadata)

The owner added a sixth clip that **does not** follow the `clip<N> <jersey>.mp4` convention (§3.1) — it is a
**pre-edited highlight compilation** ("HIGHLIGHTS JORDAN THOMAS" title card, an outro "Thank You For Watching"
card with contact info, a burned-in red arrow marking the target player in several segments, no jersey number
in the filename or run config). **Measured 2026-08-27:**

| | |
|---|---|
| Resolution / fps / duration | 3840×2160, 30 fps, 234.16 s, audio AAC 44.1 kHz stereo |
| Structure | title card (~0–3.7 s) → **14 camera takes across what look like several different matches/venues** (night turf under floodlights, a stadium with an American-football-lined field, several day grass fields with different kits/lighting/crowd size) → outro "Thank You For Watching" card (~229–234 s) |
| Cuts (`ContentDetector`, `downscale_width=640`) | **threshold-dependent, unlike clip4.** 21.0/24.0/27.0 all agree on the same **13-cut** set: t≈3.7, 25.9, 43.0, 60.4, 78.0, 90.2, 105.3, 124.9, 169.2, 180.0, 198.0, 215.1, 229.0 ⇒ **14 takes**. The existing global `configs/shots.yaml` threshold (**38.0**, tuned on clip4 — ADR-7) only finds **7** of these ⇒ would wrongly report 8 takes. **All 13 confirmed real** by eyeballing a before/after frame pair at each timestamp (same discipline as ADR-7): every one shows a genuinely distinct venue/lighting/kit change or the intro/outro card boundary — none is a within-take pan/zoom false positive. `configs/shots.yaml` now carries `jordan_thomas_highlight_video: 24.0` as a per-video override — see **ADR-16**. |
| Burned-in red arrow | present intermittently (measured via the existing `find_arrow`, `configs/detect.yaml` thresholds, 2 fps scan): **107 accepted hits** clustered in short bursts inside several takes (~t=30–33, 38–42, 47–49, 64–66, 79–82, 92–95, 107–109, 131–134, 152–155, 170–173, 182–190), not just the "opening seconds of one take" pattern seen on clip2/clip4. Same masking pipeline, no code change needed — it already generalises. |
| Jersey number | **not given anywhere in run metadata.** The intro title card (t≈1 s) *does* show a crisp, unambiguous **"22"** on a club jersey (crest reads "…SOCCER, SINCE 1988") — but that is a **promo photo, not in-game footage**, and the contact-sheet survey (42 frames @ 1-per-6s) shows the player wearing visibly **different kits/colours across different takes** (this is a highlight reel stitched from multiple matches, the same pattern as clip4's 4-venue compilation but more extreme — 8+ segments). The "22" from the title card is logged as **contextual evidence only**; it is **not** auto-applied to any in-game take. See ADR-15. |

This clip is the trigger for ADR-15 (verified-jersey identification, since there is no filename number to use
as even run metadata) and ADR-16 (cut-detection threshold no longer generalizes across a heterogeneous
compilation the way it did across the single-source §3.1 clips).

**Full pipeline run 2026-08-27 (ADR-15/16 built + executed end-to-end):** ADR-16's per-video threshold
correctly reproduced the measured 14-take/13-cut structure. ADR-15's identity verification checked all 13
non-empty takes (take 0, the 3.8s intro card, had zero crops clearing the height threshold) and **verified
zero of them** — every escalated crop got a real, working-model answer (0 VLM failures; see ADR-15's own row
in §5.1 for the fallback-model story and the evidence quotes), never a quota-silenced default. `output/
jordan_thomas_highlight_video/` therefore correctly has an empty `players/` (no folder at all), a fully
populated `identity_report.json` covering all 14 takes, and a full-resolution `original_annotated_video.mp4`
(3840×2160/30fps/234.13s, original AAC audio preserved) showing green boxes on every detected player/
official throughout and an "IDENTITY: unverified in this segment" panel + CUT banner note at every take —
exactly the honest, non-forced outcome CLAUDE.md's own accuracy rule requires when the evidence doesn't
support a claim, not a build failure.

### 3.4 Third input (pending): a SoccerNet broadcast clip, to isolate "does jersey OCR/VLM work at all
on clear footage" from "this specific amateur footage has no legible numbers" (added 2026-08-27)

§3.3's 0/14 result is well-evidenced, but it's a compound answer: it doesn't separate "the ADR-15
verification *method* doesn't work" from "this *footage* (small/blurry/side-on players, wide fixed amateur
camera) genuinely has nothing to read." Owner asked (2026-08-27) to test on **clear** footage to isolate
which. Professional broadcast (SoccerNet) is the obvious control: tight camera work, players fill much more
of the frame, HD. **Blocked on the owner completing SoccerNet's NDA form** (soccer-net.org — a Google Form
tied to the requester's own identity, not something this pipeline can submit on their behalf); `.env`'s
`SOCCERNET_PASSWORD` is empty until then. `scripts/download_soccernet.py` is built and ready (see §8) —
listing available match names needs no password (the game-list JSON ships inside the `SoccerNet` pip
package itself), only the actual video download does. Default target picked and verified real via that
listing: `england_epl/2016-2017/2016-09-24 - 14-30 Manchester United 4 - 1 Leicester`, `1_720p.mkv`,
trimmed to the first 5 minutes by default (§6's "validate on a short clip first," not a 45-minute half).

**Important scope note, not yet resolved:** SoccerNet broadcast video carries no burned-in arrow (that
graphic is specific to the owner's own Veo exports, §3.2 consequence 1) and this clip's filename won't carry
a jersey number either (same filename-less path as §3.3, ADR-15 applies) — so there is no "the target
player" for this clip, only "whichever track `heuristic_fallback_seed` locks onto." That's fine for
answering the narrow legibility question this test exists to answer, but it means this run is a capability
check on the OCR/VLM stage, not a meaningful end-to-end "did we correctly identify a named player" run —
don't over-read a verified number here as validating the *selection* heuristic, only the *reading* one.

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
| ADR-17 | **Finish real goal detection (scoreboard-delta) + best-effort scorer/assist attribution, self-gated on actually finding a legible scoreboard — never on the `profile` label** | Owner asked (2026-08-27) to "do something about goal and assist" rather than leave `detect_goals_scoreboard_delta` a permanent `NotImplementedError` stub. §3.2(3)/§3.4 established goals are a genuine **data ceiling** on the owner's own footage (no scoreboard exists), but the incoming SoccerNet broadcast clip (§3.4) is exactly the input this path was always meant for — the stub just needed finishing, not redesigning. **Scope of what's real, in order of confidence:** (1) **goal occurrence** — OCR-scan a small set of CANDIDATE scoreboard regions (top-left, top-right; score bugs are essentially never bottom-corner) across several early sampled frames, using the existing `configs/shots.yaml: ocr` EasyOCR engine; a region only "activates" for a run if it yields a STABLE, plausible `digit [sep] digit` pattern across multiple frames — this is a measured self-check, not trust in one hardcoded ROI or the `profile == broadcast` label (ADR-11's own lesson: read the measured signal). A goal event fires on a debounced digit increment in an activated region — this is the highest-confidence signal in the chain (literally reading the score). (2) **scoring team** — deliberately sidesteps ever needing to know which on-screen digit is "home"/"away": credit whichever team (our own colour-cluster, not the scoreboard's own left/right semantics) held ball possession (existing `src/events/possession.py::possession_runs_for_take`/`track_team`) in the window immediately before the digit changed. (3) **scorer** — same idea one level down: the individual identity with the last possession run before the goal, gated to the team from (2). (4) **assist** — new `EventType.ASSIST`; the rule is literally the owner's own words back: the nearest preceding `PASS` event (`src/events/possession.py::detect_passes`, which already carries `evidence["receiver_identity"]`) whose receiver is the credited scorer, from a teammate, within a configurable `assist_window_seconds` — credited to the passer. Each step down the chain (occurrence → team → scorer → assist) stacks a heuristic on a heuristic, so confidence is deliberately driven DOWN at each step and every event's `evidence` carries the full trail (which region/frames the OCR locked onto, the possession-window scores considered, which specific `Event.id` was picked as the assist) — Golden Rule 5, same "stack of flagged heuristics" pattern ADR-13 already established as owner-authorized for touches/passes/tackles. **Deliberately NOT built:** visual ball-crosses-the-goal-line/mouth detection (the owner's own "ball goes under the white box" framing). No current detector class represents a goal frame/net (`ball, player, referee, goalkeeper` only) and adding one means training a new class from scratch (Golden Rule 2); a homography-based goal-mouth-crossing check would need automatic pitch-keypoint calibration for a PANNING broadcast camera, which ADR-3 explicitly left as an unevaluated future item, not a static-camera 4-point calibration like the owner's own clips use; a naive "ball bbox disappears near a corner region" shortcut would be exactly the fragile, ungrounded guess Golden Rule 5 warns against (corners, saves, clearances, and camera cuts all look similar to that heuristic). Flagged as a candidate future ADR contingent on broadcast pitch-keypoint calibration landing first (which calibrated speed/distance would also benefit from), not silently dropped. **Still correctly "not available" on the owner's own 5 clips + `jordan_thomas_highlight_video`** — now because the self-gating OCR scan measurably finds no stable scoreboard pattern there (a live check, not a hardcoded "these clips have no scoreboard" fact repeated from §3.2(3)). | 🔧 building 2026-08-27 |
| ADR-18 | **Spec-completeness audit against the owner's full 29-section jersey/event spec (2026-08-27) found three real gaps, fixed alongside ADR-17** | Owner asked to confirm "everything I told you in the prompt" is actually wired into the pipeline/statcard, ahead of manually confirming a jersey number for the incoming SoccerNet clip (§3.4). Audited section-by-section against the actual code (not assumed) and found: **(1) no human-confirmed-identity path existed.** `target_jersey` could only ever come from the `clip<N> <jersey>` filename regex — a filename-less video (the owner's own stated new workflow: watch the clip, then tell the system the number) had no seam to receive that. Fix: `main()` gains a `--target-jersey` CLI override for filename-less videos; `src.identity.verify` threads an optional `human_confirmed_jersey` through aggregation — a human-given number needs only ONE non-contradicting read to verify (a human directly watching the footage is stronger evidence than an independent OCR/VLM multi-frame vote, Golden Rule 4), but a read that ACTIVELY DISAGREES with the human's number is surfaced, never silently suppressed (Golden Rule 5 — disagreement is itself evidence worth keeping). **(2) highlight-category clips were cut from the RAW input, not the annotated render** — found by tracing `write_player_output`'s `video_path` argument back to `run_extended_pipeline_for_video`'s own raw-file parameter; `render_full_annotated_video` didn't even run yet at the point highlights were cut. Directly contradicts owner spec §24 ("highlight videos should also show RED/GREEN boxes, ball tracking, event name, timestamp, jersey number"). Fix: reorder so the full annotated render happens BEFORE the per-player loop, and highlight-clip cutting reads from `original_annotated_video.mp4`, not the source. **(3) green-box labels read a bare `#{track_id}`** — indistinguishable at a glance from a jersey number, undermining the owner's own explicit §7 requirement ("track ID and jersey identity are different concepts... do NOT assume Track ID 10 = Jersey #10"). Fix: green boxes now read `ID: {track_id}`; the target's red box reads `#{jersey_number} | TARGET | ID: {track_id}` — both identifiers visible, neither ambiguous. **Not a gap, verified correct:** the live stats panel already shares the exact same per-player `Event` list used to write `statcard.md` (`_NumberProgress.advance_to`, keyed correctly per verified jersey number per take) — owner spec §20's "no separate hardcoded counters" rule was already satisfied, confirmed by reading the render loop rather than assumed. | 🔧 building 2026-08-27, alongside ADR-17 |
| ADR-16 | **Cut detection: fixed-threshold `ContentDetector` does not generalize across a heterogeneous multi-match compilation; per-take identity now depends on getting this right** | §3.3 measured that `Jordan Thomas Highlight Video.mp4`'s real cuts sit at content-value scores as low as ~21–27 (a 13-cut set, stable across thresholds 21.0/24.0/27.0 — the same "find the plateau" method ADR-7 used), while clip4's three confirmed **false**-positive foreground-crossings score up to 32.0. No single fixed threshold can keep clip4's false positives out *and* catch this new video's real cuts — the existing global default (38.0, tuned on clip4) misses 6 of the 13; lowering the global default to clip4's false-positive ceiling would corrupt clip4. This now matters more than it used to: ADR-15's per-take jersey verification is only as good as take segmentation — a missed cut means the tracker (and then the jersey-OCR/VLM step) runs straight across a real venue/match change, contaminating one take's "verified identity" with another game's frames. **Resolution (scoped, low-risk):** `configs/shots.yaml: scenedetect` gains an `overrides: {<video-slug>: <threshold>}` map, keyed by the same slug `work_dir_for()` already uses everywhere; `detect_takes()` looks itself up there and falls back to the existing global `threshold` when absent — so the original 5 clips' behaviour is provably unchanged (no entry for them). `jordan_thomas_highlight_video: 24.0` is the measured entry, verified by eyeballing contact-sheet frames either side of each of the 13 cut timestamps before locking it in (same verification discipline as ADR-7). PySceneDetect's `AdaptiveDetector` (same library/license, rolling-average threshold instead of one fixed number) would be the more general long-term fix for arbitrary future compilations, but is **deferred** — not evaluated here — to keep this change small and reviewable. | ✅ adopted 2026-08-27 (per-video override); AdaptiveDetector deferred |
| ADR-15 | **Verified-jersey identification required when no filename jersey number exists; no verified number ⇒ no target identity ⇒ no target stats for that take** | Owner request (2026-08-27, full spec in chat): for a video with no `clip<N> <jersey>` filename metadata, the target player's jersey number must be **read from visual evidence and confirmed**, never guessed from appearance, position, team, or the burned-in arrow alone (the arrow is a *location* prior, not an identity — §3.2 consequence 1 already says this; it reads no digits). This is a deliberate, owner-authorized pull-forward of a narrow slice of **Phase 2** (§6) for this input only — Golden Rule 7's "no jersey auto-ID until Phase 1 done" is superseded here by an explicit owner instruction, the same pattern as ADR-13's exception, not a silent scope change. **Pipeline addition:** for each **take** (cuts matter more now — ADR-16), take the arrow-hint track (existing `selection.py` logic, generalized) as a **location candidate only**; crop that track's box across the take at native/high resolution wherever it's front/back-facing and large enough to plausibly read a number; run **EasyOCR** (Apache-2.0, already in `configs/shots.yaml`'s license-approved OCR row) plus **Gemini** (owner-supplied key, ADR-14) as a cross-check on the same crops; require **temporal agreement across multiple frames**, not a single read (owner's explicit "don't randomly pick one frame's answer" rule). A take's identity record is `{take_id, jersey_number: int|None, status: verified|unverified, confidence, evidence_frames[]}`. **Unverified take ⇒ no red box, no target stats for that take** — green boxes and ball tracking still render (those don't depend on identity), and the run report says plainly "target could not be reliably identified in take N," never a guessed fallback. **Output is grouped by verified jersey number, not by an assumed single continuous human identity**: because this compilation's segments come from different matches with visibly different kits (§3.3), two takes verified to *different* numbers are **not** merged into one player just because the same arrow-selection heuristic picked a person in both — that would itself be an appearance/position-based identity assumption, which the owner explicitly forbade. Each distinct verified number gets its own `output/<slug>/players/player_<N>/` folder (statcard, highlights, timeline); the single annotated video still shows one continuous red box per take, labeled with that take's own verified number. Gemini model note: `gemini-2.5-flash` (ADR-14's implicit assumption) now 404s for new callers ("no longer available to new users") — resolved to **`gemini-3.6-flash`** (confirmed reachable 2026-08-27), with retry-on-503 (measured transient `UNAVAILABLE` during testing).
> **✅ Built + run end-to-end 2026-08-27.** Concrete details pinned during implementation: crop min height **120px** at native 4K (`configs/identity.yaml: crop.min_crop_height_px`, derived from the §3.2 "legible at 4K" finding scaled to native res); at most **6** OCR-ambiguous-or-silent crops per take escalated to the VLM (`vlm.max_escalations_per_take`, cost control mirroring ADR-14's own pattern); verification requires **>= 2 independent frames** (OCR and/or VLM, any mix) agreeing on the identical digit string, with a **strict, never-arbitrarily-broken tie rule** (`aggregation.min_agreeing_frames`/`aggregate_take_identity` in `src/identity/verify.py`) — a 2-vs-2 split, or any tie for the top count, stays unverified. Evidence collection is ONE single native-resolution decode pass over the whole video (not per-candidate-frame seeking), bucketed into takes by timestamp, specifically to avoid the keyframe-seek inaccuracy that could otherwise leak one take's frames into another's identity evidence.
> **⚠️ Mid-run discovery: `gemini-3.6-flash`'s free tier carries a hard 20-REQUESTS-PER-DAY quota** (`quotaId: "GenerateRequestsPerDayPerProjectPerModel-FreeTier"`) — the first real end-to-end run exhausted it partway through take 6 of 14, and the original single-model retry logic just retried the same exhausted model until giving up, silently turning every later take into a false "unverified" (crops never actually looked at). **Fixed**: `src/common/gemini.py::call_gemini_vision` now takes an ORDERED FALLBACK LIST (`configs/identity.yaml`/`configs/key_moments.yaml: vlm.models`: `[gemini-flash-lite-latest, gemini-3.1-flash-lite, gemini-3.6-flash]`), and distinguishes a 429 with body `error.status == "RESOURCE_EXHAUSTED"` (daily quota — skip straight to the next model, no point retrying) from a transient 429/5xx (still retried on the SAME model with backoff first). Which model actually answered is recorded in every evidence entry (`"[model=<name>] <response text>"`).
> **Real result on `Jordan Thomas Highlight Video.mp4`, re-run with the fix: 0/14 takes verified.** Every one of the 13 non-empty takes' escalated crops got a genuine answer from a working model (0 VLM failures anywhere — confirmed by inspecting `identity.json`'s own evidence trail, not just the summary counts) — e.g. *"The player is viewed from the side and no jersey number is visible"*, *"The image is extremely blurry and low-resolution"*, *"The jersey number on the back is partially visible and obscured"*. EasyOCR likewise found zero confident reads anywhere. This is a genuine, well-evidenced negative, not a quota-silenced default: the arrow-marked target specifically is too blurred/angled/small to read across every take, even in frames where a *different*, non-target player's own number is clearly legible (spot-checked visually in the rendered output at t≈65s, take 6: player #21's "9" is crisp while the arrow-marked player is unreadable from that angle). `output/jordan_thomas_highlight_video/players/` is therefore correctly EMPTY — no forced result. | ✅ built + run 2026-08-27 |
| ADR-1 | **PyAV + ffmpeg CLI for decode; drop `decord`** | `decord` 0.6.0 has no cp311/cp312 wheels and is effectively unmaintained; building it from source is a needless risk. PyAV covers seeking/frame-accurate decode, and `ffmpeg -hwaccel cuda -c:v h264_cuvid` covers NVDEC bulk decode. Same capability, zero build risk. | ✅ adopted |
| ADR-2 | **Roboflow's soccer player/pitch checkpoints are YOLOv8 → AGPL → cannot ship** | `roboflow/sports` is MIT *code*, but the released player-detection and field-keypoint **weights are Ultralytics YOLOv8**, which is AGPL-3.0. Using them violates Golden Rule 6. Resolution ladder: (a) an **Apache-2.0 RF-DETR** soccer checkpoint; (b) **RF-DETR COCO-pretrained** (`person` + `sports ball`); (c) optional overnight fine-tune. **No training in Phase 1 either way.** | ✅ **resolved → see ADR-8** |
| ADR-8 | **Phase-1 detector = `julianzu9612/RFDETR-Soccernet` (Apache-2.0), with RF-DETR-COCO as fallback** | Investigated 2026-08-24 with the validated Roboflow key. **(1)** The three canonical Roboflow soccer projects (`football-players-detection-3zvbc`, `football-field-detection-f07vi`, `football-ball-detection-rejhg`) return `model: None` on **every** version — Roboflow hosts **no trained weights** for them, only data. Their *datasets* are **CC BY 4.0** (commercially usable with attribution) — so they are excellent **eval + future fine-tune** material, which is how we use them. **(2)** On HF, `julianzu9612/RFDETR-Soccernet` is **Apache-2.0**, RF-DETR-Large (128 M, DINOv2 backbone, 1280², 1.46 GB) with exactly the classes we need — `ball, player, referee, goalkeeper` — reporting mAP@50 **0.857** / mAP **0.498** on SoccerNet. **(3)** The alternative `OrbitalLab/mova-rfdetr-soccernet-v1` (MIT) is **gated** (needs an access request) → skipped. | ✅ adopted for P1 |
| ADR-9 | **⚠️ SoccerNet-derived weights are dev-only until licensing is cleared** | ADR-8's checkpoint is *declared* Apache-2.0 by its uploader, but it was **trained on SoccerNet-Tracking-2023**, and §7 lists SoccerNet data as **research/education only**. Whether an uploader can relicense a model trained on restricted data is legally unsettled — so this is a Golden-Rule-6 "flag before adding", not a silent adoption. **Use it for Phase-1 development and evaluation** (internal R&D, not distribution). **Before any commercial ship**, take one of: (i) obtain SoccerNet commercial terms, (ii) fine-tune RF-DETR on the **CC BY 4.0** Roboflow soccer dataset, or (iii) fall back to RF-DETR-COCO. Do not ship (i)-unresolved. | ⚠️ open — revisit before ship |
| ADR-11 | **Downstream stages must consume the *motion score*, not the static/panning *label*** | Stage 0.5 put **4 of 5 clips inside the ambiguous band** (1.5–6.0 px/frame). clip2 scores **3.79 → `single-panning`** while clip4 scores **3.14 → `single-static`**, purely because the band's midpoint is 3.75 — essentially identical footage landing on opposite sides of a knife-edge. Worse, the label is contaminated by the ~3–5 s settling pan every clip opens with (§3.2), which is a startup transient, not a camera regime. Acting on the label would mean "calibrate homography once" vs "re-estimate continuously" flipping on noise. **Resolution:** the label stays in `RunProfile` for reporting, but **ADR-3's homography cadence reads `motion_score` directly** — recalibration interval scales continuously with measured motion, with the settling window excluded. For Veo footage the honest answer is that the camera digitally pans to follow play, so homography needs periodic re-estimation *regardless* of which label it got. | ✅ adopted — implement at Stage 2/pitch |
| ADR-14 | **Full extended-output spec: exact statcard.md format, full-resolution annotated original video, per-category highlight reels, Gemini for celebrations** | Owner supplied a complete, detailed spec (2026-08-25) superseding/extending ADR-13 — see **§13** for the full text. Key decisions: (1) the "primary output" is the **original video at original resolution/fps/duration/audio**, annotated in place (red box = target, green = others, ball marker, live stat panel) — not a separate downscaled debug render; (2) `statcard.md` must use the owner's exact template with real computed values, `[uncertain]` wherever a stat can't be reliably derived rather than any invented number (owner's own accuracy rule, independently identical to Golden Rule 5); (3) category-specific highlight compilations (`ball_possession.mp4`, `dribbles.mp4`, `assists.mp4`, `goals.mp4`) — an empty file (not a fabricated one) when a category never occurs, per the owner's own "do not fabricate a highlight" rule; (4) celebrations/"other key moments" — asked the owner directly (no heuristic exists that means anything here); owner chose to supply a **Gemini API key** rather than a cheap heuristic or skipping — Gemini is used ONLY to classify short, cheaply-pre-filtered candidate windows as celebration/key-moment or not; it never touches player/ball detection, which stays RF-DETR (Golden Rule 1 — pipeline of specialized models, VLM is one more specialized stage, not a replacement). | ✅ owner-authorized 2026-08-25 |
| ADR-13 | **Owner explicitly authorized best-effort touch/pass/tackle/save heuristics ahead of Phase 1 completion** | Golden Rule 7 reserves fine-grained action stats for Phase 2+ and requires asking the owner before breaking it. Asked (2026-08-25): owner chose "best-effort heuristics now" over staying Phase-1-only or building a real fine-tuned action-spotting model. Scope: touch/pass/tackle/save-for-GK detectors, **no training, no new pretrained model** (Golden Rule 2 still holds) — proximity/possession-change/speed heuristics on top of existing detections+tracks, each with a low, honestly-stated confidence ceiling (Golden Rule 5). **Still structurally impossible regardless of this decision:** goals and assists on this footage — no scoreboard exists to verify a goal (§3.2(3)), so assist's own rule ("the pass before a goal") can never fire here. **Superseded by ADR-14** on celebration/key-moment scope (Gemini approved, see above). | ✅ owner-authorized 2026-08-25 |
| ADR-12 | **Team assignment = torso-colour clustering, NOT SigLIP, on this footage** | §5 Stage 3 recommended SigLIP→UMAP→KMeans (the `roboflow/sports` recipe). Measured on `clip2`, it failed twice: team splits of **48/3** then **27/2/18-NaN** where a real two-team split is ~50/50 — it was clustering grass and pose, not kit. A controlled experiment on the *same* tracks (torso band y∈[0.15,0.50]·h, x inset 18%, boxes ≥25 px, mean **CIELAB**, KMeans k=2, 39 usable tracks) gave **24/15** with cluster means L\*=**97** (dark-blue kit) vs L\*=**146** (white kit) — a large, physically meaningful separation. **Why SigLIP loses:** it is trained for image–*text* semantic alignment, so at 60–150 px crop size every crop reads as "a soccer player" and the dominant variance is pose/blur/background, not jersey colour. It is right for Roboflow's large broadcast crops, wrong for youth Veo footage at this scale. **Resolution:** `team.method: colour` (default) with SigLIP retained as a config-selectable fallback for real broadcast input. **Cluster per take, not globally** — clip4 spans 4 venues with very different lighting. Confidence = distance to own centroid relative to inter-centroid distance. | ✅ adopted |
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
| `requests` (ADR-15, `src/identity/jersey_vlm.py`'s Gemini HTTP calls) | Apache-2.0 | ✅ |
| `SoccerNet` (pip package, `scripts/download_soccernet.py`) | MIT | ✅ package code only — the video *data* it downloads is the separate, NDA-gated row below |
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
scripts/download_soccernet.py   # --list (no password needed) or pulls one broadcast clip into
                                 # input/ (needs SOCCERNET_PASSWORD in .env, from the NDA form at
                                 # soccer-net.org) -- used 2026-08-27 to get a clear-broadcast test
                                 # case for ADR-15's jersey-verification pipeline, see §3.4
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
❌ Train from scratch. ❌ Ship YOLO/BoxMOT (AGPL). ❌ Auto-ID player in Phase 1. ❌ Track across cuts.
❌ Emit untraceable stats. ❌ Build everything at once. ❌ Hard-code params. ❌ Over-promise accuracy in
UI copy. ❌ Feed raw 4K to a detector.
> ❌ ~~Pass/touch/tackle stats in Phase 1~~ — **owner-authorized exception, ADR-13/14, §13.** Best-effort
> heuristics only, never a fine-tuned model, always a low honestly-stated confidence.
> ❌ ~~Auto-ID player in Phase 1~~ — **owner-authorized exception, ADR-15, §3.3, for inputs with no filename
> jersey number.** Identity must still be *verified* (OCR + VLM temporal agreement on visible digits), never
> inferred from appearance/position/team/the arrow alone; unverified ⇒ no red box, no target stats for that
> take. This does not relax Golden Rule 7 for the original §3.1 clips, which keep filename-metadata-only,
> human-selects-the-track Phase 1 behaviour.

## 13. Extended output spec (owner-authorized 2026-08-25 — ADR-14, revised 2026-08-27 — ADR-15;
supersedes prior output shape)

> **ADR-15 gating (2026-08-27):** on an input with **no filename jersey number** (§3.3), everything in this
> section that names "the target player" is conditioned on that take's identity being **verified** (jersey
> number read from visible evidence with temporal agreement — ADR-15), never on the arrow/track-selection
> heuristic alone. An unverified take still gets full green-box + ball-tracking output; it just contributes
> no red box and no target-player stats. On the original §3.1 clips (filename gives the number), nothing in
> this section changes — the human-confirms-the-track Phase-1 flow is untouched.

### 13.1 Primary output: full-resolution annotated original video
**`output/<slug>/original_annotated_video.mp4`** — the original input video, unchanged resolution/fps/
duration, **original audio preserved**, with detections/tracking/events/stats drawn directly on top.
Not a separate downscaled debug clip — the actual footage the owner uploaded. Draw:
- **Red** box, thick, persistent label, around the **verified target player**, labeled with that take's own
  verified jersey number (e.g. `#22 | TARGET`) — never a bare track ID standing in for a jersey identity
  (ADR-15: a tracking ID and a jersey number are different concepts, never conflated). ID must not visibly
  reset when the target overlaps another player or leaves/re-enters frame **within the same take**
  (best-effort continuity via the existing within-take stitching logic, generalised to run continuously
  rather than gated to arrow-hint votes only; this is still inference on top of raw tracks, not magic — flag
  it as approximate, never silently drop it). A take with no verified identity gets **no red box at all** —
  every detected player in that take renders green, plus a visible "IDENTITY: unverified in this segment"
  note alongside that take's "CUT — take N" banner.
- **Green** boxes, thin, around every other detected player, with a track ID where available.
- A small distinct **ball marker** (already exists from Stage 2's SAHI ball detection) — solid when
  observed, visually distinct (e.g. hollow/dashed) when interpolated across a gap (Golden Rule 5: never
  present an inferred position as if it were observed).
- A **live stats panel** (semi-transparent, corner-anchored) that accumulates as the video plays: every
  category in §13.3, each showing its real running count/value, or a static "not detected"/"uncertain" line
  for whatever the current build genuinely cannot produce — never a fabricated or frozen-fake number.
- A brief "CUT — take N" banner at every take boundary (existing pattern from `scripts/overlay_tracks.py`).
4K NVENC encode + audio mux is expensive (multi-minute per clip, large files) — that is an accepted cost,
not a reason to substitute a downscaled render.

### 13.2 `statcard.md` — exact template (owner-specified, **revised 2026-08-27 — ADR-15**)
The owner restated the template on 2026-08-27 with an explicit **Identity Status** line and a **confidence
column** on the timeline — this supersedes the plain-text 2026-08-25 version. One file per **verified**
jersey number (§13.5), never one file blending two different verified numbers:
```markdown
# Player Statistics

## Player #<jersey_number>

**Identity Status:** Verified

**Touches:** [actual value]
**Passes:** [actual value]
**Sprints/Runs:** [actual value]
**Goals:** [actual value]
**Assists:** [actual value]
**Shots:** [actual value]
**Tackles:** [actual value]
**Saves:** [actual value]
**Dribbles:** [actual value]
**Possession Time:** [actual value]
**Distance Covered:** [actual value]

## Event Timeline

| Time | Event | Confidence |
|------|-------|------------|
| [Timestamp] | Touch | [0-1] |
| [Timestamp] | Pass | [0-1] |
| [Timestamp] | Sprint | [0-1] |
| [Timestamp] | Dribble | [0-1] |
| [Timestamp] | Shot | [0-1] |
| [Timestamp] | Tackle | [0-1] |
| [Timestamp] | Assist | [0-1] |
| [Timestamp] | Goal | [0-1] |
| [Timestamp] | Celebration | [0-1] |
| [Timestamp] | Other Key Moment | [0-1] |
```
Only list events that actually fired; **never pad the timeline to look complete.** Wherever a value can't be
reliably derived, write `uncertain` (owner's own words) rather than a number — Golden Rule 5 in the owner's
own terms, not a new rule. `Distance Covered` and any speed-derived value are additionally suffixed
`(uncalibrated)` per ADR-6 — never bare metres/km. `Goals`/`Assists` read `not available` when no scoreboard
exists to verify against (§3.2(3), true for the original §3.1 clips; re-check per-video for new input — don't
assume it transfers). If a take's identity is **unverified** (ADR-15), no `statcard.md` is written for it at
all — the run report states plainly which takes/segments had no verified target and why, per the owner's own
words: *"Target player could not be reliably identified. Statistics were not generated because jersey
identity could not be verified."*

### 13.3 Event categories and their real status on this footage
| Category | Status | Method |
|---|---|---|
| Sprint, Shot | ✅ built (Stage 4/5, pre-ADR-13) | speed/direction heuristic |
| Touch, Pass, Tackle, Save (GK) | ✅ built + wired 2026-08-27 (ADR-13) | ball-proximity / possession-change heuristic, no training, low confidence ceiling; computed per take by `src/events/aggregate.py::compute_take_all_events`, attributed to one verified player by `attribute_events_to_target` |
| Dribble, Ball-possession, Possession duration | ✅ built + wired 2026-08-27 (ADR-14) | extension of the same possession-heuristic layer |
| Distance covered | ✅ built + wired 2026-08-27 (ADR-14) | same normalised-pixel-unit speed integration as sprints (ADR-6) — always `(uncalibrated)`; summed ONE TAKE AT A TIME per verified jersey number (raw track ids reset per take, Golden Rule 3) |
| Celebration, Other key moment | ✅ built + wired 2026-08-27 (ADR-14) | cheap motion pre-filter (`src/events/key_moments.py::find_candidate_windows`, excludes windows already explained by a counted sprint/dribble/possession) → **Gemini** (same model-fallback list as ADR-15's identity check) classifies only the surviving candidate windows, emitting an event ONLY on an explicit "yes"; never runs on every frame (cost control). Not exercised on `Jordan Thomas Highlight Video.mp4` (0 verified players -- no player timeline existed to search for key moments against). |
| Goal, Assist | 🔧 building (ADR-17) | scoreboard-OCR delta for occurrence (self-gated on actually finding a legible scoreboard, not the `profile` label); team/scorer via last-possession-before-goal; assist = nearest preceding teammate `PASS` to the scorer. Still reports `not available` on any input where the self-gating scan finds no stable scoreboard (true for every clip in `input/` today, §3.2(3)/§3.4) — a measured result, not a hardcoded one |

### 13.4 Highlight compilations (per clip, category-specific — not the single ranked reel)
`output/<slug>/players/player_<N>/highlights/{ball_possession,dribbles,assists,goals,key_moments}.mp4` —
each a concatenation of every event in that category for **that verified player**, a few seconds of context
before/after each event (config knob, matches the existing `pre_seconds`/`post_seconds` pattern). **An
empty/absent file, not a fabricated one, when a category has zero real events** (owner's rule, identical in
spirit to Golden Rule 5). Each highlight clip itself carries the same red/green/ball overlay plus an
event-name + timestamp + jersey-number caption (owner spec §24) — it is a cut of the annotated video, not the
raw footage.

### 13.5 Required directory layout (per input video)
**When the filename gives the jersey number (§3.1 clips, Phase-1 human-confirms-the-track flow):** unchanged,
one flat `output/<slug>/` per clip as before.

**When the jersey number is not in the filename and must be verified (ADR-15, §3.3):** identity is
per-*verified-number*, not per-video — a compilation can legitimately contain more than one, or none:
```
output/<slug>/
  original_annotated_video.mp4     # §13.1 — ONE file, whole video, red box only in verified takes
  players/
    player_<N>/                    # one folder per DISTINCT verified jersey number found
      statcard.md                  # §13.2 — exact owner template, this player's real values only
      highlights/
        ball_possession.mp4
        dribbles.mp4
        assists.mp4                # expect empty on most amateur footage (no scoreboard, §3.2(3))
        goals.mp4                  # expect empty on most amateur footage (no scoreboard, §3.2(3))
        key_moments.mp4            # celebrations / other key moments, Gemini-classified (§13.3)
      events/
        event_timeline.json        # machine-readable form of statcard.md's timeline, incl. confidence
  identity_report.json             # every take's {take_id, jersey_number|null, status, confidence,
                                    # evidence_frames} — including UNVERIFIED takes, so "why is there no
                                    # player_7 folder for take 3" is always answerable from this file
```
The pre-existing `reel.mp4` / `stat_card.json` / `stat_card.md` / `run_report.json` / `clips/` artifacts are
kept alongside (other tooling/tests depend on them) — §13.5 is additive, not a replacement of the verified
Stage 4–6 output.
