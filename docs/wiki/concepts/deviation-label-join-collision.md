---
type: concept
tags: [dataset, labels, kaggle, training, gotcha]
code_refs: [scripts/convlstm_mobilenetv1_all_frames.py, scripts/mobilenetv1_two_head_all_frames.py, scripts/make_kaggle_allframes_dataset.py]
updated: 2026-08-16
---

# Deviation label join collision

Two families of two-head trainers pair the `obstacle_freepath` and
`deviation_lrn` manifests of the [[kaggle-image-datasets]] all-frames build
([[kaggle-two-head-dataset]] is a different pipeline). They join on **different
keys**, and only one of the two keys is unique. The non-unique join silently
relabels **4,537 `left`/`right` frames as `none`**.

## The two joins

| trainer family | join key | unique? |
|---|---|---|
| `*_two_head_all_frames.py` › `_pair_task_dirs()` | `(source_clip, filename)` | yes |
| `convlstm_mobilenetv1_all_frames.py` › `read_lateral_labels()` | `(source_clip, frame_index)` | **no** |

`read_lateral_labels()` builds `{(source_clip, frame_index): label}` in a dict
comprehension, so when a key repeats the **last row wins**.

## Why the key collides

`make_kaggle_allframes_dataset.py` merges three sources into one scheme and
**strips the `_head`/`_tail` suffix from `source_clip`** on purpose, so an
endpoint cut and the labelled cut of the same physical recording group together
under the trainer's clip-based split (see [[kaggle-image-datasets]] and
[[exports-output]]). That is correct for splitting, but it means two different
images — one `cuts` frame and one `endpoints` frame of the same recording — can
share the same `(source_clip, frame_index)` pair. `endpoints` frames are always
labelled `free_path`/`none`, so they overwrite the real `left`/`right` label of
the cut frame.

Measured on `data/derived/kaggle_allframes_datasets_224/deviation_lrn/labels.csv`
(24,847 rows, sources `cuts` 16,584 / `endpoints` 6,523 / `clear` 1,740):

- 18,324 unique keys, **5,334 colliding keys**, 4,537 of them disagreeing on the label
- last-wins relabels **2,267 `left` → `none`** and **2,270 `right` → `none`**

## Effect on the non-test pool

Same 490 non-test clips, same 48,749 frames, deviation head only:

| join | left | right | none |
|---|---|---|---|
| `(source_clip, filename)` — correct | 4,154 | 4,253 | 40,342 |
| `(source_clip, frame_index)` — last-wins | 2,163 | 2,291 | 44,295 |

The obstacle head is unaffected (11,238 / 37,511 either way): the obstacle label
is read from the primary manifest and never joined.

**The test split is unaffected.** The `endpoints` source contains no day-22 or
day-24 clips, so no test frame has a colliding key — both joins yield
obstacle 2,230 / free_path 1,740 and left 1,247 / right 983 / none 1,740.

## What is affected

- **`sense/treino/labels.txt` on Drive** (`sensecv_allframes_dataset_v1`, 48,749
  rows). Its `dataset_info.json` names
  `convlstm_mobilenetv1_all_frames.py` as `source_trainer`, and replaying the
  last-wins join reproduces its counts exactly (2,163 / 2,291 / 44,295).
  `sense/teste/labels.txt` is correct.
- **Anything consuming `dataset_used/frames.csv`**, which the same datagen
  writes: `mobilenetv1_two_head_random_splits_kaggle.py` and its 10 FPS
  companion read `lateral_label` straight from that export.

Not affected: `mobilenetv1_/mobilenetv4_/resnet50_/efficientnetb0_two_head_all_frames.py`,
whose `_pair_task_dirs()` join is unique. The `sibgrapi` run statistics in
`anamariabarrosnonato/ss2ere` show the correct 4,154 / 4,253 non-test totals.

## Fix

Join on `(source_clip, filename)` — filenames are unique inside a task folder —
or key on `(source_clip, source, frame_index)` to keep `cuts` and `endpoints`
rows apart. A cheap assertion catches regressions:

```python
keys = [(r["source_clip"], r["frame_index"]) for r in rows]
assert len(keys) == len(set(keys)), "deviation labels.csv key is not unique"
```

## Related

- [[kaggle-image-datasets]] — the all-frames builder and its `source_clip` rule
- [[classification-taxonomy]] — what `left`/`right`/`none` mean
- [[exports-output]] — the endpoints build that supplies the colliding rows
