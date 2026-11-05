---
type: concept
tags: [experiment, state, resume, paper]
code_refs: [scripts/train_mobilenetv4_sense_kaggle.py, scripts/eval_sense_matrix.py]
updated: 2026-08-17
---

# Sense experiments: state and where to resume

Snapshot of the 2026-08-16/17 campaign so a fresh session can pick it up
without re-deriving anything. Details live in [[sense-style-datasets]],
[[allframes-split-provenance]], [[per-head-background-reduction]],
[[mobilenetv4-sense-experiments]] and [[dinov2-sense-finetuning]].

## Where everything lives

| what | path |
|---|---|
| datasets | `C:\Users\Isaque\Desktop\sense_run4_*_224\` (4 folders) |
| results, checkpoints, DINOv2 selections | `C:\Users\Isaque\Desktop\sense_run4_experiments\` |
| build / eval / trainer scripts | `scripts/build_sense_*.py`, `scripts/eval_sense_*.py`, `scripts/train_*_sense_*_kaggle.py` |
| run4 reference statistics | `…\sense_run4_experiments\referencia\sibgrapi_run_stats\` |

Nothing needed lives in a session scratch directory any more.

**Kaggle** (account `anamariabarrosnonato`, all private):
dataset `sensecv-run4-sense` (919 MB, both tars);
kernels `sensecv-mobilenetv4-two-head`, `sensecv-mobilenetv4-5-fps`,
`sensecv-mobilenetv4-dinov2-0-98`,
`sensecv-mobilenetv4-held-out-day-validation`,
`sensecv-dinov2-5-fps-uniao`, `sensecv-dinov2-5-fps-uniao-last-1`.

## Leaderboard (joint accuracy)

T1 = original distinct test (3,970). T4 = T1 + all janela test frames (6,676).

| model | T1 | T4 |
|---|---|---|
| MobileNetV4 full_rate/uniao | **0.9156** | 0.8607 |
| DINOv2 ultimos1 (5 FPS uniao) | 0.8962 | **0.9194** |
| DINOv2 ultimos2 (5 FPS uniao) | 0.8854 | 0.8833 |
| MobileNetV4 full_rate/baseline | 0.8806 | 0.7972 |
| MobileNetV4 dia_validacao/baseline | 0.8710 | 0.7904 |
| MobileNetV4 fps5/baseline | 0.8103 | 0.7915 |
| MobileNetV4 dino098/uniao | 0.8068 | 0.7323 |
| MobileNetV4 dia_validacao/uniao | 0.7856 | 0.6718 |
| MobileNetV4 dino098/baseline | 0.7307 | 0.5891 |
| MobileNetV4 fps5/uniao | 0.5861 | 0.5255 |
| DINOv2 completo (undertrained) | 0.4426 | 0.2681 |

## The three findings that matter for the paper

1. **Clip-grouped validation is not measuring generalization.** Validation loss
   0.0003 with clip-grouped clips versus 0.275 for a genuinely held-out day
   (21-06-2026) on the same task. Validation clips come from the same corridors
   as training clips.
2. **The union's advantage does not survive an honest validation split.** Under
   clip-grouped validation uniao beats baseline (0.9156 vs 0.8806); under
   held-out-day validation it loses (0.7856 vs 0.8710).
3. **The paper's split table mixes two different quantities** — Train/Val are
   post-per-head-reduction (frames counted twice), Test is the raw natural
   split. See [[allframes-split-provenance]].

## Open threads

- **DINOv2 `completo` at `lr 1e-5`** — the only way to say anything about full
  fine-tuning; the current run stopped mid-convergence at run4's `lr 1e-4`.
- **Clean union ablation** — train on the union but validate on the distinct
  validation set only, so baseline and uniao early-stop under identical
  criteria. Every `uniao` number so far is confounded by a different
  validation set.
- **Pre-cut versus post-cut scoring on the janela test set** — 899 of its 2,706
  frames are `pos_corte=1`, where the maneuver is over but the label still says
  `left`/`right`. Splitting the metric would separate model error from the
  labelling assumption.
- **The paper's exact table triple (33,188 / 6,102 / 3,970)** matches no
  archived run; run4 is 33,118 / 6,172 / 3,970. Either rerun and quote the new
  artifact or repoint the table at run4.
- **`docs/wiki` DINOv2 trainers** (`train_dinov2_classifier.py` and friends)
  predate this campaign and use the older per-head/endpoint policies; they were
  not touched.

## Gotchas that cost time

- Kaggle batch GPU sessions get a **Tesla P100 (sm_60)**; the preinstalled torch
  is built for sm_70+. The trainers detect this, install `torch==2.5.1+cu121`
  and re-exec once. The `accelerator` metadata field is ignored by the API.
- A kernel's slug comes from its **title**, not `id`; a mismatch returns 409.
- Only **2** concurrent batch GPU sessions.
- `kaggle kernels output` serves the last *completed* version's log, so a
  running kernel exposes no partial log — poll status instead.
- DINOv2 selection is **not** a per-frame property: the same threshold over a
  different pool keeps a different subset (953 distinct frames alone versus
  1,382 of the same frames when janela is in the pool).

## Related

- [[mobilenetv4-sense-experiments]]
- [[dinov2-sense-finetuning]]
- [[sense-style-datasets]]
- [[allframes-split-provenance]]
- [[deviation-label-join-collision]]
