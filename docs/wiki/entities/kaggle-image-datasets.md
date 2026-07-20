---
type: entity
tags: [dataset, kaggle, training, mobilenet, dinov2]
code_refs: [scripts/make_kaggle_image_datasets.py, scripts/train_mobilenet_datasets.py, scripts/train_mobilenet_dino.py, scripts/make_kaggle_allframes_dataset.py, scripts/train_mobilenet_allframes.py, scripts/predict_test_frames.py]
updated: 2026-07-14
---

# Kaggle image datasets (single-head)

Two **separate** single-task image datasets (not the [[kaggle-two-head-dataset]])
built straight from the `datasetcortado/deviation` cut folders, plus a MobileNetV1
trainer for them. Unlike the two-head pipeline these scripts are **self-contained**:
they read only the cut folders (`cut_info.json` + the already-trimmed `<clip>.mp4`)
and need no `sensecv.app`, IMU JSON, or `history.json`, so they run on Kaggle from
the uploaded `deviation` folder with just OpenCV (+ PyTorch for the DINOv2 variants,
TensorFlow for training).

## Source: the `deviation` cut folders

Produced by [[crop-suggestion]]'s deviation cut and exported via
`make_datasetcortado.py` / `cut_manifest_deviation.py`. Each leaf folder holds a
`cut_info.json` and a `<clip>.mp4` that is **already trimmed** to the cut window
(~1 s, ~29-30 frames, 720×720). `cut_info.json` carries `event_type` ∈
{`desvio`, `parada`, `livre`} and `side` ∈ {`LEFT`, `RIGHT`, `NONE`} (see
[[classification-taxonomy]] and [[imu-event-labeling]]). Folders without a
`cut_info.json` are skipped.

## Builder — `scripts/make_kaggle_image_datasets.py`

Decodes each cut's frames once and writes **four ImageFolder-style datasets**
(class sub-folders + `labels.csv` + `dataset_info.json`):

| dataset | task | classes | frames kept |
|---|---|---|---|
| `obstacle_freepath` | obstacle vs free path | `obstacle`, `free_path` | all (stride) |
| `deviation_lrn` | deviation direction | `left`, `right`, `none` | all (stride) |
| `obstacle_freepath_dino` | same | same | DINOv2-selected |
| `deviation_lrn_dino` | same | same | DINOv2-selected |

Label mapping (project taxonomy):

| event_type | side | obstacle label | deviation label |
|---|---|---|---|
| desvio | LEFT | obstacle | left |
| desvio | RIGHT | obstacle | right |
| parada | NONE | obstacle | none |
| livre | NONE | free_path | none |

`labels.csv` columns: `filename,label,event_type,side,source_clip,frame_index,time_s`.
Datasets 1 & 2 share the same frames under two label schemes; 3 & 4 share the
DINOv2-selected subset. Each folder is self-contained (zip + upload on its own).

**DINOv2 selection** (`Dinov2Selector`): embeds each clip's frames with DINOv2
ViT-S/14 (torch.hub `facebookresearch/dinov2`, transformers `facebook/dinov2-small`
fallback) and keeps a diverse subset — a frame is kept when its cosine similarity
to the last kept frame drops below `--dino-threshold` (0.98, the same consecutive-
frame metric as [[clear-dataset]]), capped at `--dino-max-per-clip` (8). Needs
Kaggle Internet ON to download the weights; `--no-dino` skips datasets 3 & 4.

## Trainer — `scripts/train_mobilenet_datasets.py`

Trains one **MobileNetV1, α=0.5, 128×128, ImageNet weights** per dataset
(`MobileNet(include_top=False, pooling='avg')` → Dropout → softmax head sized to
the dataset's classes). Mirrors the [[kaggle-two-head-dataset]] trainer's two-phase
schedule: frozen-base warmup (lr 1e-3) then whole-net fine-tune (lr 1e-4), best by
`val_loss`, early stopping. Generic over 2-class and 3-class datasets.

Key choices:

- **Temporal split by clip.** Splits over the `source_clip` column, never over
  frames, so a clip's frames never span two splits (no leakage). **Test = the
  fixed `--test-day` collection (day-of-month, default 22)** — every day-22 clip,
  imbalance ignored. Train/val = the remaining clips, split **stratified by class**
  (`--val-frac` 0.15 of each class's non-test clips → val), so both see every
  class. The day is parsed from the clip name's `DD-MM-YYYY` (same `_DATE_RE` idea
  as `train_two_head_mobilenet.py`).
- **Class balance** via inverse-frequency sample weights computed on the train
  frames (e.g. free_path ≪ obstacle), plus `weighted_metrics`.
- **Flip safety.** Horizontal-flip augmentation is auto-**disabled** for
  `deviation_lrn*` (classes `left`/`right`) because a flip would invert the label;
  it stays on for `obstacle_freepath*`. Brightness/contrast jitter is used for both.

Outputs per dataset under `--out-root`: `model.keras`, `report.json` (confusion
matrix + per-class P/R/F1 + the held-out `test_day`), `history.json`, and a
top-level `summary.json`.

`scripts/train_mobilenet_dino.py` is a **standalone build+train pipeline**
(no imports of the other scripts — paste-into-a-cell friendly) over the
**all-frames image datasets** (see below): `--input` is the uploaded
`kaggle_allframes_datasets_224` root (default
`/kaggle/input/datasets/isaqueefraim/dataset8/kaggle_allframes_datasets_224`).
It first **builds `obstacle_freepath_dino` + `deviation_lrn_dino` itself** —
no video decoding: it groups the obstacle scheme's `labels.csv` rows by
physical frame sequence (the filename base before `__fNNNNN.jpg`, so head/tail
endpoint cuts stay separate sequences), runs the same `Dinov2Selector`
cosine-dedup per sequence, and copies the kept frames into both schemes
(matched by filename), preserving all columns incl. `source`/`segment` —
then runs the same trainer over them at **img_size 224** (the dataset's native
size; `--out-root` defaults to `models_dino/`). The build is skipped
automatically when both `labels.csv` already exist (`--rebuild` forces it;
`--no-build` skips it outright). Use it to train MobileNet on **only** the
DINOv2-selected frames for a like-for-like comparison against
`train_mobilenet_allframes.py`'s full-frame models. Same split / balance /
flip-safety / embedded-Rescaling / canonical-class-order logic.

As of 2026-07-14, the DINO trainer runs a small **fine-tuning strategy sweep**
(`--trials`, default `baseline,last_block,clip_balanced`) on the exact same
clip-grouped split and seed:

| trial | fine-tuning | regularization / balance |
|---|---|---|
| `baseline` | whole base, lr `1e-4` | Adam, dropout 0.20, inverse-frequency weights, basic jitter |
| `last_block` | last 30 base layers, lr `3e-5`; BatchNorm frozen | AdamW, dropout 0.35, stronger label-safe crop/color jitter |
| `clip_balanced` | last 45 base layers, lr `5e-5`; BatchNorm frozen | AdamW, dropout 0.30, equal class then equal clip influence, stronger jitter |

Every trial restores its **global best `val_loss` checkpoint across both
phases**; the prior implementation checkpointed the best epoch but then
overwrote it with the final epoch. Validation and test datasets now use unit
weights, so train-only balancing does not redefine held-out metrics. Reports
add macro-F1 and balanced accuracy; this matters for `deviation_lrn_dino`,
where high `none` recall can hide weak `left` recall in overall accuracy.

Trial artifacts live at `<out-root>/<dataset>/trials/<trial>/`. The promoted
`<dataset>/model.keras` is selected by **validation macro-F1** (validation
accuracy and loss only break ties); day-22 test metrics are recorded for every
trial but never participate in selection. `<dataset>/comparison.csv` provides
a flat side-by-side table of parameters and validation/test metrics. The
selected report remains at the old `<dataset>/report.json` path for
viewer/prediction-script compatibility.

The saved `model.keras` is **drop-in usable in the live viewer**, by two design
choices:

- **Embedded preprocessing.** The model's first layer is
  `Rescaling(1/127.5, -1)`, so it takes **raw 0-255 RGB** pixels and normalises
  internally — the same contract `app.py › _sensemodel_preprocess` feeds (it
  assumes the model rescales). The data pipeline therefore feeds raw 0-255 too
  (augmentation on the 0-255 scale), **not** `mobilenet.preprocess_input`. Input
  size is read from the model, so 128×128 is picked up automatically.
- **Canonical class order.** Classes are emitted as `free_path,obstacle` (width-2)
  and `left,right,none` (width-3), matching `SENSECV_OBSTACLE_LABELS`
  /`SENSECV_DEVIATION_LABELS`. The viewer (`/api/sensemodel`, `/api/activation`)
  auto-detects a single-head model by output width and labels it correctly — see
  [[activation-maps]]. Upload one through the viewer's model picker and only its
  head (obstacle **or** deviation) lights up; the other shows `—` / "cabeça
  ausente".

## Running on Kaggle

Both scripts are notebook-friendly: paste into a cell and edit the top-of-file
`CONFIG` dict, or run as a script (CLI flags override `CONFIG`); a notebook
kernel's injected args are ignored via `parse_known_args`.

```python
# 1) build the four datasets (Internet ON for DINOv2)
!python make_kaggle_image_datasets.py \
    --input /kaggle/input/datasets/isaqueefraim/datasetandre/deviation \
    --out   /kaggle/working/datasets
# 2) train MobileNetV1 on each, test = day 22
!python train_mobilenet_datasets.py \
    --datasets-root /kaggle/working/datasets \
    --out-root /kaggle/working/models --epochs 40
```

## Combined all-frames variant — `scripts/make_kaggle_allframes_dataset.py`

A second builder (2026-07-01) that merges **three sources** into the same two
ImageFolder schemes (`obstacle_freepath` + `deviation_lrn`, every decodable
frame, no DINOv2 subset):

| source (`labels.csv` column) | input | labels |
|---|---|---|
| `cuts` | `datasetcortado-andre/datasetcortado/deviation` labelled cut folders | per `event_type`/`side` (same table as above) |
| `endpoints` | `manifest_exports/dataset_endpoints/deviation` — first/last 0.5 s of every clip ([[exports-output]] endpoints build; no event label) | `free_path` / `none` |
| `clear` | `data/test-clear` — `NN_sem_obstaculos` clear clips, **the day-22 free-path recordings** (cf. [[clear-dataset]]) | `free_path` / `none` |

Same self-contained contract (OpenCV only, notebook `CONFIG`, CLI overrides).
Differences from the single-source builder:

- `labels.csv` adds `source` (cuts/endpoints/clear) and `segment` (head/tail)
  columns so any source can be filtered out downstream.
- `source_clip` **strips the `_head`/`_tail` suffix**, so an endpoint cut and
  the labelled cut of the same physical clip group together under the trainer's
  clip-based split (no cross-split leakage from one recording).
- **`sources.csv` label fallback.** A cut folder without a per-folder
  `cut_info.json` is labelled from the dataset root's `sources.csv` ledger
  (`output_folder` → `event_type`/`side`). This matters: 136 of the 396 andre
  folders — including 23 of the 33 day-22 clips — have no `cut_info.json` and
  would otherwise silently vanish, leaving the day-22 test split nearly empty.
- **Clear clips are date-tagged.** `--clear-date` (default `22-06-2026`)
  prefixes `SenseCV-<date>-clear_` onto the clear clips' `source_clip` so the
  trainer's `DD-MM-YYYY` day parser places them: they ARE day-22 recordings
  and form the free-path side of the day-22 test set.
- Root-level `review*.mp4` files are excluded (the endpoints root carries a
  `review_all_frames.mp4` montage that must not be ingested), and an endpoints
  folder missing both `cut_info.json` and a ledger row is still used (segment
  inferred from the folder suffix).

Frames are resized to **224×224×3** on write (`--img-size`, default 224;
`0` keeps the native 720×720 — the sources are square so no distortion).

Current build (2026-07-01, output `data/derived/kaggle_allframes_datasets_224/`):
893 clips (396 cuts + 439 endpoints + 58 clear, 0 skipped) → **24 847 images
per scheme** at 224×224; obstacle_freepath = 11 894 free_path / 12 953 obstacle;
deviation_lrn = 13 802 none / 5 619 left / 5 426 right; ~0.9 GB total. Under
`train_mobilenet_datasets.py`'s day-22 split the test set is 91 clips
(33 cuts + 58 clear) = 3 156 frames: obstacle 1 416 / free_path 1 740;
deviation left 850 / right 566 / none 1 740. Known gap: the `endpoints` source
itself holds **no day-22 clips** (it predates the 22-06-2026 UFC uploads), so
day-22 endpoint frames would need a rebuild via `make_dataset_endpoints.py`.

**Trainer:** `scripts/train_mobilenet_allframes.py` — a standalone copy of the
trainer (same family as `train_mobilenet_dino.py`) whose `CONFIG` defaults to
the uploaded all-frames dataset root (`obstacle_freepath,deviation_lrn`, no
build step — trains straight from `/kaggle/input`), **img_size 224** (the
dataset's native size), and `models_allframes/` output. Same clip-grouped
day-22 split / class weights / flip safety / embedded Rescaling / canonical
class order as the other trainers; the report adds `test_frames_by_source`.
Split verified locally: train 308 clips / val 55 / test 91 (3 156 frames,
cuts 1 416 + clear 1 740) for the obstacle scheme.

## Test-set prediction dump — `scripts/predict_test_frames.py`

Standalone post-training companion (same notebook `CONFIG` + CLI contract).
For every dataset with a trained `<models-root>/<dataset>/model.keras` it
rebuilds the trainer's test set (every frame whose `source_clip` day-of-month
== `--test-day`; the test split depends only on the day, so it reproduces
without the trainer's seed), predicts all test frames (raw 0-255 in — the
models embed Rescaling), and writes a browsable dump per dataset under
`--out`: one folder **per predicted label** holding a copy of each test frame
(correct frames named `ok__<name>`, mistakes `true-<true_label>__<name>`), a
`predictions.csv` (true/pred labels, confidence, per-class probabilities), and
a `summary.json` (accuracy + per-true-label recall and `predicted_as`
breakdown). Class order comes from the model dir's `report.json`, falling back
to the canonical width-2/width-3 orders. `CONFIG` defaults target the
`_dino` datasets + `models_dino/`; pass `--datasets ""` to sweep every dataset
that has a model.

## Relation to the two-head pipeline

This is the **single-head** alternative to [[kaggle-two-head-dataset]]: two
independent classifiers (obstacle, deviation) instead of one network with two
heads, built directly from the deviation cuts rather than manifest-export lateral
cuts + the [[clear-dataset]]. The DINOv2 dedup follows the [[clear-dataset]]
selection convention.
