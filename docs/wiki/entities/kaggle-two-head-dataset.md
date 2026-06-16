---
type: entity
tags: [dataset, kaggle, training, two-head]
code_refs: [scripts/make_kaggle_two_head_dataset.py, scripts/train_two_head_kaggle.py, scripts/kaggle_trainval_copy_paste.py]
updated: 2026-06-11
---

# Kaggle two-head dataset

`data/derived/kaggle_two_head_dataset/` — the image-classification dataset
uploaded to Kaggle (`isaqueefraim/dataset5`) to train the two-head MobileNetV2
model (`obstacle_head` + `deviation_head`). Built by
`scripts/make_kaggle_two_head_dataset.py`; consumed by
`scripts/train_two_head_kaggle.py` (random stratified split) and
`scripts/kaggle_trainval_copy_paste.py` (per-clip temporal train/val split,
paste-into-notebook variant).

## Classes

| head | classes |
|---|---|
| `obstacle_class` | 0 = clear / no obstacle, 1 = obstacle |
| `deviation_class` | 0 = left (esquerda), 1 = right (direita), 2 = none |

## Layout

```
kaggle_two_head_dataset/
  labels.txt               # "file_name obstacle_class deviation_class"
  labels_with_source.tsv   # + source_dataset, source_label
  class_map.json           # counts, class maps, per-clip extraction summary
  dataset/
    obstacle__<dataset>__<clip-folder>__frame_NNNNN.jpg
    clear__<clip>__frame_NNNNN.jpg
```

## Sources and labeling

- **Obstacle images** come from the manifest-export lateral cuts
  (`data/derived/manifest_exports/<dataset>/lateral/`, see [[export-pipeline]]
  and [[crop-suggestion]]). Frames are decoded from each clip MP4 and
  de-duplicated with grayscale SSIM (`_ssim_gray` from [[app-backend]],
  threshold 0.97, max gap 0.5 s, first/last always kept). `deviation_class`
  comes from the `deviation_side` column of each `sources.csv`
  (esquerda → 0, direita → 1, empty/parada → 2); `obstacle_class` is always 1.
- **Clear images** are copied verbatim from `data/derived/clear/images/`
  (see [[clear-dataset]]). That folder now includes the original
  `*_sem_obstaculos` export frames plus the repaired SSIM 0.95 final-second
  uploaded-video supplement, all labeled `0 2`.

## Current build (2026-06-11)

Built from the zero-threshold lateral export (223 clips across 4 uploaded
datasets — see [[crop-suggestion]] for the threshold history):

| joint (obstacle_deviation) | images |
|---|---|
| 1_0 (obstacle, left) | 1978 |
| 1_1 (obstacle, right) | 1874 |
| 1_2 (obstacle, none/parada) | 155 |