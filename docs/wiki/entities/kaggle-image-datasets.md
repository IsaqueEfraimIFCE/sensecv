---
type: entity
tags: [dataset, kaggle, training, mobilenet, dinov2]
code_refs: [scripts/make_kaggle_image_datasets.py, scripts/make_kaggle_exports_dataset.py, scripts/train_mobilenet_datasets.py, scripts/train_mobilenet_dino.py, scripts/train_mobilenetv1_endpoints_kaggle.py, scripts/train_mobilenetv1_selected_multilabel_kaggle.py, scripts/train_mobilenetv1_allframes_merged_kaggle.py, scripts/train_dinov2_classifier.py, scripts/train_dinov2_endpoints_kaggle.py, scripts/save_dinov2_endpoints_train_dataset.py, scripts/evaluate_dinov2_test.py, scripts/make_kaggle_allframes_dataset.py, scripts/train_mobilenet_allframes.py, scripts/predict_test_frames.py]
updated: 2026-07-28
---

# Kaggle image datasets (single-head)

Two **separate** single-task image datasets (not the [[kaggle-two-head-dataset]])
built straight from the `datasetcortado/deviation` cut folders, plus a MobileNetV1
trainer for them. Unlike the two-head pipeline these scripts are **self-contained**:
they read only the cut folders (`cut_info.json` + the already-trimmed `<clip>.mp4`)
and need no `sensecv.app`, IMU JSON, or `history.json`, so they run on Kaggle from
the uploaded `deviation` folder with just OpenCV (+ PyTorch for DINOv2/V4 and
TensorFlow for V1–V3 training).

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
(no imports of the other scripts — paste-into-a-cell friendly). Its `--input`
is the uploaded `Todos_cortes_auto_trim/deviation` tree: each child folder has
a trimmed MP4 and label metadata in `cut_info.json`, with root `sources.csv` as
fallback. It decodes each video (`--stride`), runs DINOv2 cosine selection per
clip, and writes the shared selected frames into `obstacle_freepath_dino` and
`deviation_lrn_dino`. It then reads `--day22-clear-dir` (default: dataset8's
`kaggle_allframes_datasets_224/deviation_lrn/none`) and sends all **1,740**
`*sem_obstaculos*.jpg` source images through the same DINOv2 selector, grouped
independently by their 58 clear `source_clip` values. Only the selected subset
(threshold/cap controlled by the normal `--dino-threshold` and
`--dino-max-per-clip`) enters the outputs as `free_path` and `none`. Existing
`labels.csv` provenance is preserved, including `SenseCV-22-06-2026-*` clip
values. Outputs rebuild automatically when their `dataset_info.json` does not
match the source-image count or DINO selection settings (`--no-build` still
overrides this). The held-out
collection consequently defaults to day 22, now containing both the labelled
obstacle/direction cuts and clear `free_path`/`none` examples.

The backbone is selected by `--mobilenet-version v1|v2|v3|v4`. Every choice is
fixed to baseline width (`--alpha 1.0`) and 224×224; other values are rejected,
and pretrained-weight failures stop the run instead of silently training from
scratch:

| choice | pretrained backbone | backend / promoted artifact |
|---|---|---|
| `v1` | Keras MobileNetV1 ImageNet, α=1.0 | TensorFlow/Keras, `model.keras` |
| `v2` | Keras MobileNetV2 ImageNet, α=1.0 | TensorFlow/Keras, `model.keras` |
| `v3` | Keras MobileNetV3Large ImageNet, α=1.0 | TensorFlow/Keras, `model.keras` |
| `v4` | timm `mobilenetv4_conv_small.e2400_r224_in1k` | PyTorch/timm, `model.pt` |

V1–V3 retain embedded `Rescaling(1/127.5,-1)` and are drop-in compatible with
the live viewer. TensorFlow 2.17 has no MobileNetV4 application, so V4 uses
timm's ImageNet-normalized 224px Conv-Small checkpoint while retaining the same
clip split, weighting, two-phase trials, validation-only selection, reports,
and canonical class order. Install `timm` in Kaggle if needed
(`!pip install -q timm`). Its `.pt` checkpoint is not directly loadable by the
viewer's current `.keras` route.

## DINOv2 classifier — `scripts/train_dinov2_classifier.py`

This companion pipeline uses DINOv2 for the classifier itself, rather than only
for frame selection. It delegates dataset construction to
`train_mobilenet_dino.py`, then adds a dropout + linear classification head to
pretrained DINOv2 ViT-S/14 (default) or ViT-B/14 (`--dino-model vits14|vitb14`).
Official `torch.hub` weights are preferred, with Hugging Face Transformers as a
fallback. The promoted artifact is a PyTorch `model.pt` checkpoint containing
the class order, input size, normalization, and backbone identifier.

The classifier requires the labelled-MP4/day-22-clear builder contract from the
current `train_mobilenet_dino.py`; it rejects the older image-folder builder so
the same output folder can never be populated from the wrong source pipeline.
It first searches notebook attachments for a matching companion, then falls
back to an integrity-checked bundled copy of that exact builder. Consequently,
`train_dinov2_classifier.py` can be uploaded or pasted by itself without
downloading `train_mobilenet_dino.py`. The shared build defaults are cosine
threshold `0.96`, at most 25 selected frames per clip, and seed `43`; the
MobileNet script defaults to its `clip_balanced` trial.

The trainer retains the project contracts: 224×224 ImageNet normalization,
clip-grouped train/validation/test splits, fixed day-22 test data, canonical
class order, class/clip weighting, and no horizontal flip for left/right labels.
Its default validation-only sweep compares a frozen linear probe, last-block
fine-tuning, and last-two-block fine-tuning. Each trial checkpoints the global
best validation loss; validation macro-F1 chooses the promoted model. The
current Flask viewer only loads Keras models, so this `.pt` artifact requires a
PyTorch inference integration before it can be selected in the viewer.

`scripts/evaluate_dinov2_test.py` is the standalone post-training evaluator for
the promoted DINOv2 checkpoints. It reconstructs the fixed test split directly
from `labels.csv` (`source_clip` day-of-month 22 by default), restores each
checkpoint's backbone, class order, image size, and normalization, and performs
inference without augmentation or balancing weights. It writes `metrics.json`
and `predictions.csv` per dataset plus a combined `summary.json`; metrics include
unweighted cross-entropy, accuracy, macro-F1, balanced accuracy, per-class
precision/recall/F1/support, and the confusion matrix.

### All-frame Kaggle linear probe

`scripts/train_dinov2_endpoints_kaggle.py` is a deliberately smaller,
single-cell alternative for one all-frames dataset at a time. It has one
editable `CONFIG` block and no CLI parser, `main()`, strategy sweep, or class
weighting. The legacy filename says `endpoints`, but the trainer starts from
every eligible frame before split-specific background sampling.

Before frame loading, the trainer excludes the manifest clips whose
`OBSTACULO` cell is `CAIXA` plus a qualifier (not a plain `CAIXA`): IFCE
02-06 IDs 09–28, IFCE 06-06 IDs 10–12/30–32/50–52/70–72, and IFCE 10-06 IDs
05–07/20–22/35–37/50–52. The list is embedded because the Kaggle input ships
`labels.csv`, not the original `.xlsx` manifests. `metrics.json` records the
excluded clip list and source-frame count.

Every valid image row from the remaining clips is retained. Dataset partitioning
still uses the established policy: all frames from one `source_clip` stay
together, collection days 22 and 24 are the fixed test set, and the remaining
clips are split into class-stratified train/validation groups by clip count.
Collection days are derived only from an explicit `DD-MM-YYYY` inside
`source_clip`. There is no fallback based on `sem_obstaculos`, `source=clear`,
the image filename, or its folder; undated rows therefore do not enter the
fixed day-22/day-24 test set.

For a dataset containing both `left` and `right`, the balanced training split
is doubled with horizontal-flip copies: `left` becomes `right`, `right` becomes
`left`, and `none` remains `none`. The original validation and test frames are
never augmented.
`CONFIG["left_right_flip_augmentation"]` controls the behavior and defaults to
enabled; obstacle/free-path training skips it because those classes do not
contain left/right. The checkpoint and `metrics.json` record source, flipped,
and effective training-frame counts.

The reviewed 24-07 UFC package is merged before splitting and is required by
default. `CONFIG["additional_dataset_roots"]` defaults to `["auto"]`, which
finds `kaggle_24-07-2026_ufc_exports_unique_224` beside the local project or
anywhere under `/kaggle/input`; an explicit package root or task directory is
also accepted. `CONFIG["test_days"] = [22, 24]` holds out both collection dates:
the original 3,156 day-22 frames and all 814 July-24 frames are test-only.
`CONFIG["excluded_days"] = [9]` removes every day-9 clip entirely before
splitting; day-9 auto-trim sources are skipped before video decoding.
Overlapping `source_clip` values are skipped rather than duplicated, and
`require_july24_data=True` prevents a Kaggle run from silently omitting the
attachment. The checkpoint and metrics preserve the resolved paths,
accepted/skipped frame/clip counts, and both test days.

The trainer also builds a task-specific all-frame dataset from
`manifest_exports/Todos_cortes_auto_trim/deviation` under
`<output_dir>/built_trainval_auto_trim/<task>/`. Before decoding, it compares
the auto-trim `source_clip` names and full `DD-MM-YYYY` dates against the base
all-frames plus July-24 package. Existing clips and every clip from an already
loaded date are ignored, as are all 44 known CAIXA clips. This leaves 171 new
clips from 02/03/05/06/07 July and 29,124 frames for the train/validation pool;
day 9 is excluded, while days 22 and 24 remain test-only. Labels come from
`cut_info.json`, falling back to the root `sources.csv`. The built folder
includes `labels.csv` and
`dataset_info.json`, and its rows join the normal clip-grouped stratified split
before training augmentation.

No class balancing or frame-rate reduction is applied after the clip-grouped
split: every eligible frame is retained in train, validation, and the natural
day-22/day-24 test set. Training-only horizontal-flip augmentation runs after
splitting; validation and test are not augmented. The checkpoint and
`metrics.json` record source and effective training-frame counts, but contain
no background-balance metadata.

Pretrained DINOv2 is frozen and used once to cache CLS embeddings. Training
updates one `torch.nn.Linear` layer from the 384-dimensional ViT-S/14 embedding
to the two- or three-neuron class output. Early stopping and best-state
selection use validation loss, while the test set remains report-only. Output
is limited to `model.pt` (linear-layer state plus backbone and preprocessing
metadata) and `metrics.json`. Console output includes the compact dataset
summary and final validation/test result. The script is intended to run on
Kaggle with Internet enabled and is not a viewer-loadable checkpoint by itself.

#### Legacy endpoint training subset

`scripts/save_dinov2_endpoints_train_dataset.py` materializes only the training
rows from the former endpoint/split policy. It independently reproduces the
first/last sequence selection, undated-clear day-22 mapping, fixed test day,
clip-grouped stratified validation split, seed, and validation fraction. It
does not reproduce either the current all-frame selection or its compound-CAIXA
manifest exclusion.

The output preserves the original class-relative image paths and adds
`labels.csv` with clip, sequence, endpoint, frame, time, and source provenance.
`split_summary.json` records the train/validation/test frame, clip, and class
counts even though only train images are copied. Existing output is protected
unless `CONFIG["overwrite"]` is explicitly enabled.

#### All-frame train_val/test materializer

`scripts/save_dinov2_allframes_trainval_test_dataset.py` materializes the
current trainer input pool without making a train/validation split. It is
standalone and inlines the ingestion rules for base data, reviewed July-24
rows, auto-trim discovery, compound-CAIXA exclusion, and day-9 exclusion, so it
can be pasted directly into one Kaggle cell.

The output is multi-label rather than two duplicated task trees:
`train_val/images/` and `test/images/` each contain every physical image once.
The root `labels.csv` assigns two targets to every row:
`obstacle_label` (`free_path`/`obstacle`) and `deviation_label`
(`left`/`right`/`none`), along with split, source clip, and source path.
Base and July-24 task manifests are joined one-to-one by
`(source_clip, frame filename)` and fail if either label is missing.

Explicitly dated days 22 and 24 enter test; every other eligible original
frame enters train_val. No balancing, frame subsampling, or augmentation is
materialized. Auto-trim is decoded only once: obstacle and deviation targets
are both derived from its event metadata. `split_summary.json` records unique
image counts plus both label distributions and source/exclusion details. The
temporary decode cache is removed after copying, and no trainer script is
required at runtime.

#### Universal DINOv2 cosine selector

`scripts/select_dinov2_universal_similarity_kaggle.py` is a standalone Kaggle
selector for the multi-label materialized dataset. It reads the first 1,000
valid `train_val` manifest rows, extracts normalized frozen DINOv2 ViT-S/14 CLS
embeddings, and processes those candidates in manifest order. The first frame
is accepted; every later train_val frame is accepted only if its cosine
similarity is strictly below 0.90 against every already selected train_val
frame. This greedy universal rule guarantees that every selected train_val pair
is below the threshold. Test rows bypass DINOv2 completely and are all copied
unchanged.

Selected images remain unique and retain `obstacle_label`,
`deviation_label`, split, and source-clip provenance. `labels.csv` contains the
selected train_val subset plus the complete original test split.
`selection_audit.csv` records train_val acceptance/rejection and nearest
selected neighbors, and `selection_summary.json` records configuration,
selected/rejected train_val counts, preserved test count, and the independently
verified maximum train_val pairwise similarity.
`dataset_dir="auto"` discovers a compatible two-label manifest under
`/kaggle/input`; multiple matches require an explicit path.

#### Selected-image DINOv2 classifiers

`scripts/train_dinov2_selected_multilabel_kaggle.py` trains directly from the
universal selector output and never loads an unselected frame. Manifest
`split=test` membership is authoritative, unchanged, and was never filtered by
DINOv2. Only selected
`train_val` rows are divided into clip-grouped, class-stratified train and
validation subsets using the established 0.15 clip-count policy and seed 43.

Frozen DINOv2 embeddings are shared, then one single `torch.nn.Linear` head is
trained for `obstacle_label` and another for `deviation_label`. The obstacle
head uses original selected training images. The deviation head retains the
existing training-only horizontal-flip policy (`left`/`right` swap and `none`
remains `none`); validation and test are never augmented. Outputs are
`obstacle_freepath.pt`, `deviation_lrn.pt`, and combined `metrics.json`, which
records selected train/validation/test frames, clips, labels, and held-out
metrics. Console output prints every class count for source train, effective
post-augmentation train, validation, and test for each classifier.

Background balancing runs independently on train and validation after the clip
split. It preserves every obstacle/left/right row while choosing evenly spaced
frames within each `free_path` or `none` clip at one proportional keep rate.
Within each split, `free_path` targets the obstacle-frame total and `none`
targets the rounded mean of left and right. The largest-remainder allocation
makes each split's global target exact while per-clip center-bin selection
approximates a fixed lower frame rate. Test remains natural and untouched.

Checkpoint and metrics output include train and validation before/after class
totals, global keep fractions, and per-clip selected counts, keep fractions,
and frame strides. Console output prints before-balance and balanced counts for
both splits, post-augmentation training counts, and untouched test counts. No
class weights or weighted sampler are used.

#### Selected-image MobileNetV1 classifiers

`scripts/train_mobilenetv1_selected_multilabel_kaggle.py` applies the same
selected manifest, authoritative test membership, per-task clip split,
independent train/validation background balancing, and deviation flip policy
as the selected-image DINOv2 trainer. It does not run DINOv2 or perform another
similarity selection.

Unlike the frozen DINOv2 linear heads, it initializes two independent timm
`mobilenetv1_100.ra4_e3600_r224_in1k` models from ImageNet weights and
fine-tunes each network end-to-end: one for `obstacle_label` and one for
`deviation_label`. It saves `obstacle_freepath.pt`, `deviation_lrn.pt`, and
`metrics.json` under
`/kaggle/working/mobilenetv1_selected_classifiers`. Checkpoints record the
architecture, class order, normalization, source dataset, augmentation, and
train/validation sampling summaries. Test remains unbalanced and unaugmented.

#### Merged all-frame MobileNetV1 classifier

`scripts/train_mobilenetv1_allframes_merged_kaggle.py` is the MobileNetV1
counterpart to the supplied merged all-frame DINOv2 script. It builds one
configured task directly from the base all-frame package, reviewed July-24
package, and new auto-trim clips. It uses the same compound-CAIXA filtering,
duplicate-day/clip avoidance, days 22/24 test assignment, 0.20 clip-stratified
validation split, independent train/validation background sampling, and
training-only left/right flip augmentation.

The script replaces only frozen DINOv2 embedding extraction and the linear
head: timm `mobilenetv1_100.ra4_e3600_r224_in1k` starts from ImageNet weights
and is fine-tuned end-to-end. The best validation-loss state is restored before
test evaluation and saved as `model.pt`; `metrics.json` contains the build,
split, sampling, augmentation, epoch-history, validation, and test records.
The decoded auto-trim audit dataset remains under
`built_trainval_auto_trim/<task>`.

### MobileNetV1 endpoint initialization comparison

`scripts/train_mobilenetv1_endpoints_kaggle.py` retains the former first/last
endpoint selection and clip split policy, including the rule that undated
`sem_obstaculos` / `source=clear` clips belong to day 22; it does not use the
all-frame selection above. It trains the same timm MobileNetV1 architecture
end-to-end twice on the same loaders: once from ImageNet weights and once from
random initialization.

Each run restores its best validation-loss state before the fixed test
evaluation. The script prints per-class train/validation/test frame counts,
saves `mobilenetv1_imagenet.pt` and `mobilenetv1_scratch.pt`, and writes both
runs, histories, split counts, and metrics to `metrics_comparison.json`.

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
`<dataset>/model.keras` (V1–V3) or `<dataset>/model.pt` (V4) is selected by
**validation macro-F1** (validation
accuracy and loss only break ties); held-out day-22 test metrics are recorded
for every trial but never participate in selection. `<dataset>/comparison.csv` provides
a flat side-by-side table of parameters and validation/test metrics. The
selected report remains at the old `<dataset>/report.json` path for
viewer/prediction-script compatibility.

The V1–V3 saved `model.keras` is **drop-in usable in the live viewer**, by two
design choices:

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

### Reviewed-export batch builder

`scripts/make_kaggle_exports_dataset.py` builds the same two ImageFolder/CSV
schemes directly from reviewed `data/exports` cuts. Export sequence numbers are
not treated as raw IDs: the builder recovers each raw clip through an exact
sequential accelerometer-sample match, then validates obstacle position and
deviation direction against the collection `.xlsx` manifest. Every decoded
export frame is resized to 224×224. No raw endpoint, unused raw clip, clear, or
synthetic frame is added.

Repeated exports of one raw clip are ordered by export-folder number and only
the first is retained; later copies are listed in `ignored_exports.json`.
`provenance.csv` preserves the export-to-raw-ID/manifest mapping, while each
dataset's `labels.csv` uses a shared date-bearing `source_clip` for all frames
from one raw clip.

The 24-07-2026 UFC build at
`data/derived/kaggle_24-07-2026_ufc_exports_unique_224/` uses 27 unique reviewed
exports (raw IDs 03, 13, and 28 have no used export). It contains 814 images per
scheme: obstacle/free-path = 814 obstacle / 0 free-path; deviation = 397 left /
417 right / 0 none. Exports 05, 08, 13, and 18 are ignored as later appearances
of raw clips 05, 07, 11, and 16 respectively. The sibling `.zip` is the
Kaggle-uploadable package.

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
