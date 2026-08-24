# Model card — RF-DETR SoccerNet (Phase-1 player/ball detector)

**Status:** ✅ adopted for Phase 1 (`CLAUDE.md` **ADR-8**) · ⚠️ **dev/eval only, not shippable** (**ADR-9**)

## Identity
| | |
|---|---|
| Source | https://huggingface.co/julianzu9612/RFDETR-Soccernet |
| File | `models/rfdetr-soccernet/checkpoint_best_regular.pth` |
| Size | 1.46 GB |
| SHA-256 | `b9ade4bcc2316259582674ebeebbd27bb2e956480428c54114ace567237eb5f9` |
| Downloaded | 2026-08-24 |
| Declared licence | **Apache-2.0** (by uploader) |

## Architecture
- **RF-DETR-Large**, 128 M parameters, **DINOv2** backbone
- Input **1280 × 1280**
- Classes (index → name): `0 ball`, `1 player`, `2 referee`, `3 goalkeeper`
- These match `DetectionClass` in `src/common/types.py` exactly — no remapping needed.

## Reported performance (publisher's numbers, on SoccerNet)
| Metric | Value |
|---|---|
| mAP@50 | 0.857 |
| mAP@75 | 0.520 |
| mAP | 0.498 |

Training: SoccerNet-Tracking-2023, 42,750 images, **4 epochs**, batch 4, lr 1e-4, A100 40 GB, ~14 h.

> ⚠️ **Do not quote these numbers as our performance.** See **ADR-10**. They were measured on
> professional broadcast footage. Our input (`CLAUDE.md` §3.2) is youth/amateur **Veo** footage: higher,
> wider fixed camera, far smaller players (~60 px at 4K), different kits, and in `clip1` American-football
> lines painted over the pitch. The real number must come from our own labeled slice (§9.1).
> Also note only **4 epochs** were completed — this is a lightly-trained checkpoint.

## Licence risk — the blocking issue
The uploader declares Apache-2.0, but the weights were **trained on SoccerNet-Tracking-2023**, which
`CLAUDE.md` §7 lists as **research/education only**. Whether an uploader may relicense a model trained on
restricted data is legally unsettled, so per Golden Rule 6 this is flagged, not silently adopted.

- ✅ **Permitted now:** internal Phase-1 development and evaluation (no distribution).
- ⛔ **Blocked:** shipping in a commercial product until resolved by **one** of:
  1. obtaining SoccerNet commercial terms;
  2. fine-tuning RF-DETR on the **CC BY 4.0** Roboflow dataset (`data/eval/roboflow-football-players-v20/`,
     v20 is already prepared as "rf-detr-m") — the cleanest exit;
  3. falling back to **RF-DETR COCO-pretrained** (`person` + `sports ball`), fully permissive but weaker classes.

See `ATTRIBUTION.md`.

## Operational notes (A2000 12 GB)
- Heavier than the RF-DETR-base/small that `CLAUDE.md` §11 recommends. Run **fp16, batch 1**.
- Expect roughly 3–4 GB VRAM at 1280²; the §11.1 budget is **5 GB** because an unrelated process holds
  6 GB of the card. **Measure peak VRAM in the Stage 2 smoke test** — if it does not fit, drop to
  RF-DETR-COCO rather than lowering input size below ~1024 (far players are already only ~60 px).
- Native 1280² input vs. our 1920-wide decoded frames: verify the resize path preserves small-object recall;
  consider SAHI tiling for **players** too, not just the ball (§3.2 consequence 5).

## Fallback
`RF-DETR-COCO` (Apache-2.0, no SoccerNet provenance) — `person` + `sports ball`. Use if VRAM fails,
if domain-gap accuracy is unusable, or as the licence-clean shipping path.
