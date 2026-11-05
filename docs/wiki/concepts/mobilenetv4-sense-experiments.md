---
type: concept
tags: [training, experiment, mobilenetv4, dinov2, kaggle, paper]
code_refs: [scripts/mobilenetv4_two_head_all_frames.py]
updated: 2026-08-17
---

# MobileNetV4 sense experiments (2026-08-16)

A campaign of two-head trainings on the [[sense-style-datasets]], run on Kaggle
GPU under the user's account. Every run uses **run4's configuration** with one
deliberate change.

## Shared configuration

timm `mobilenetv4_conv_small.e2400_r224_in1k` pretrained, 224x224,
`Resize(BICUBIC)` + `ToTensor` + ImageNet `Normalize`, batch 32, AdamW
`lr 1e-4` / `wd 1e-4`, `ReduceLROnPlateau(min, 0.5, patience 2, min_lr 1e-7)`,
fp16 AMP, 40 epochs, early stop patience 6, best state by validation loss,
seed 43, deterministic hflip on train only (`left <-> right` swapped).

**The change:** no per-head masking. Every frame carries both labels and trains
both heads — `loss = CE(obstacle) + CE(deviation)` — so
[[per-head-background-reduction]] is not applied at training time.

Two configurations per variant:

- `baseline` — distinct frames only (24,347 train)
- `uniao` — distinct + extended windows (34,394 train)

## The four variants

| variant | kernel | what changes |
|---|---|---|
| full rate | `sensecv-mobilenetv4-two-head` | nothing (reference) |
| fps5 | `sensecv-mobilenetv4-5-fps` | every split subsampled to 5 FPS (source 30) |
| dino098 | `sensecv-mobilenetv4-dinov2-0-98` | per-clip DINOv2 ViT-S/14 cosine selection at 0.98, every split |
| dia_validacao | `sensecv-mobilenetv4-held-out-day-validation` | validation = the whole 21-06-2026 collection day |

`21-06-2026` was chosen because it is the only day covering all four label
combinations with real support (free/none 814, obstacle/none 954, left 1,082,
right 1,063; 3,913 frames, 9.6% of the pool). Every other candidate day is
deviation-only or clear-only.

## Results

Joint accuracy, all eight models on six test sets. T1–T3 are distinct frames
only; T4–T6 add the janela test frames.

| model | T1 orig 3970 | T2 fps5 704 | T3 dino 953 | T4 orig 6676 | T5 fps5 819 | T6 dino 2901 |
|---|---|---|---|---|---|---|
| full_rate/uniao | **0.9156** | **0.9105** | **0.8730** | **0.8607** | **0.9011** | **0.8370** |
| full_rate/baseline | 0.8806 | 0.8807 | 0.8447 | 0.7972 | 0.8474 | 0.7308 |
| dia_validacao/baseline | 0.8710 | 0.8679 | 0.8164 | 0.7904 | 0.8510 | 0.7070 |
| fps5/baseline | 0.8103 | 0.8125 | 0.8258 | 0.7915 | 0.8132 | 0.8204 |
| dino098/uniao | 0.8068 | 0.8111 | 0.7398 | 0.7323 | 0.7973 | 0.6925 |
| dia_validacao/uniao | 0.7856 | 0.7955 | 0.7041 | 0.6718 | 0.7448 | 0.5970 |
| dino098/baseline | 0.7307 | 0.7287 | 0.6275 | 0.5891 | 0.6911 | 0.4405 |
| fps5/uniao | 0.5861 | 0.5724 | 0.4837 | 0.5255 | 0.5726 | 0.4467 |

Per-class F1, support and confusion matrices for every cell:
`C:\Users\Isaque\Desktop\sense_run4_experiments\resultados\eval_matrix.json`.
Checkpoints for all eight models are in that folder's `checkpoints/`.
A DINOv2 comparison on the same fps5/uniao data is in
[[dinov2-sense-finetuning]]; current standings in [[sense-experiments-state]].

## Findings

- **The ranking is stable across all six views.** Absolute joint accuracy moves
  by up to 15 points depending on the test set, but the ordering barely changes.
  A number quoted without naming its test set is meaningless; a *comparison*
  between models is robust.
- **T2 ~ T1 within 0.005 for every model.** Evaluating on a 5 FPS subsample of
  the test set (704 frames) ranks models identically to the full 3,970. DINOv2
  0.98 does not have this property: T3 shifts scores 3–6 points and reorders
  pairs.
- **Both subsampling strategies cost accuracy, DINOv2 more than 5 FPS**, despite
  DINOv2 keeping nearly twice as many training frames (11,057 vs 6,237).
  Selection removes the most redundant frames, which are the clear/`none` ones,
  starving the majority class the model calibrates against.
- **DINOv2 selection is not a per-frame property.** Selecting over the distinct
  set keeps 953 frames; selecting over distinct+janela keeps 1,382 of the same
  distinct frames, because the greedy rule compares against the last kept frame
  and the neighbourhood changes. Any "DINOv2-selected dataset" must state the
  pool it was selected from.
- **Deviation errors are almost never inversions.** On the janela test set
  full_rate/uniao confuses left with right on 6 of 2,706 frames (0.2%); three
  models have zero inversions. Errors are overwhelmingly a fallback to `none` —
  the benign failure mode for assistive navigation.
- **fps5/uniao collapsed** (deviation F1 0.58) with best epoch 2 of 8: early
  stopping fired before the deviation head converged, on a training set 4x
  smaller. Its obstacle score is the best on the janela set, which is an
  artifact — that set has no `free_path` frames, so obstacle "accuracy" is pure
  recall and rewards a bias toward predicting obstacle.

### The validation finding

`dia_validacao` gives validation loss **0.275** (baseline) and **1.186**
(uniao), against **0.0003** for the clip-grouped split on the same task —
roughly 1000x higher. Clip-grouped validation was not measuring generalization:
validation clips come from the same corridors and sessions as training clips.
This is direct evidence, on the project's own data, for the leakage concern the
paper raises about frame-level splits.

Consequence: **the union's advantage does not survive an honest validation
split.** `dia_validacao/uniao` (0.7856) falls below `dia_validacao/baseline`
(0.8710), reversing the ordering seen under clip-grouped validation. Its best
epoch was 1 of 7.

## Caveats

- The `uniao` runs early-stop against a validation set that includes janela
  frames, so baseline and uniao are not selected under identical criteria. A
  clean ablation would train on the union but validate on the distinct
  validation set only; not yet run.
- T3 here is 953 frames against the kernel's 950 — the selection was recomputed
  locally in CPU fp32 versus the kernel's GPU fp16, flipping ~3 borderline
  cosines. It is a faithful reconstruction, not the identical set.
- The janela test set has no `free_path` and no `none` frames, so macro-F1 over
  all classes is meaningless there; report `macro_f1_presentes` or per-class F1.
- Residual `none` predictions on janela frames are partly label noise: 899 of
  its 2,706 test frames are post-cut (`pos_corte=1`), where the maneuver is over
  but the label still says `left`/`right`.

## Infrastructure notes

- Kaggle assigns a **Tesla P100 (sm_60)** to batch GPU sessions, while its
  preinstalled torch 2.10+cu128 supports sm_70+. Scripts detect this and install
  `torch==2.5.1+cu121` then re-exec once. The `accelerator` field in
  `kernel-metadata.json` is ignored by the kernels API.
- A kernel's slug comes from its **title**, not the `id` field; a mismatch
  returns 409.
- Batch GPU sessions are capped at 2 concurrent.

## Related

- [[sense-experiments-state]] — leaderboard, artifact paths, open threads
- [[dinov2-sense-finetuning]] — DINOv2 on the same fps5/uniao data
- [[sense-style-datasets]] — the data these ran on
- [[allframes-split-provenance]] — where the splits come from
- [[per-head-background-reduction]] — the scheme these runs drop
- [[deviation-label-join-collision]]
