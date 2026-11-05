---
type: concept
tags: [dataset, split, training, run4, paper]
code_refs: [scripts/mobilenetv4_two_head_all_frames.py, scripts/mobilenetv1_two_head_all_frames.py]
updated: 2026-08-16
---

# All-frames split provenance

How the frame counts in the WiP paper's split table are produced, traced from
the Kaggle inputs to the reported numbers and reproduced locally on 2026-08-16.

## The pool

`load_allframe_rows()` merges three Kaggle attachments into one row per frame,
each row carrying **both** labels (joined on `(source_clip, filename)` — see
[[deviation-label-join-collision]] for why the key matters):

| source | slug | frames | clips |
|---|---|---|---|
| `kaggle_allframes_datasets_224` | `isaqueefraim/dataset8` | 22,781 | 410 |
| `kaggle_24-07-2026_ufc_exports_unique_224` | `isaqueefraim/dataset9` | 814 | 27 |
| `Todos_cortes_auto_trim/deviation` (decoded) | `isaqueefraim/datasetandrefinal` | 29,124 | 171 |
| **pool** | | **52,719** | **608** |

dataset8 starts at 24,847 rows; 2,066 are dropped as compound-CAIXA clips
(02-06 ids 9–28, 06-06 {10,11,12,30,31,32,50,51,52,70,71,72}, 10-06
{5,6,7,20,21,22,35,36,37,50,51,52}). Auto-trim contributes only clips whose
`DD-MM-YYYY` is absent from the base pool and is not a test day, which leaves
the 02/03/05/06/07-07-2026 recordings.

## The split

`grouped_split()`: test = every clip whose day-of-month is in
`test_days = [22, 24]`; the remainder is split **by clip**, stratified on each
clip's majority obstacle label.

**Test = 3,970 frames / 118 clips** — free 1,740 / obstacle 2,230, left 1,247 /
right 983 / none 1,740. It decomposes as 3,156 day-22 frames from dataset8
(1,740 clear + 1,416 obstacle) plus 814 day-24 UFC frames, and it is *natural*:
no reduction is applied to test.

## Why train/val in the paper are not frame counts

`make_head_balanced_rows()` runs two independent background reductions —
`free_path` down to the obstacle count for the obstacle head, `none` down to
left+right for the deviation head — then emits each surviving row **tagged with
the head whose loss it trains**. The reported count is the union of both heads'
rows, so a frame kept by both is counted twice. See
[[per-head-background-reduction]].

That is why both heads' subtotals are equal in the paper's table, and why
`obstacle` slightly exceeds `free_path` after reduction.

## Reproduced numbers (run4, MobileNetV1)

Against `anamariabarrosnonato/ss2ere` → `sibgrapi/run4/final_statistics1.txt`,
every field matches:

| stage | frames | clips | obstacle | deviation |
|---|---|---|---|---|
| train_source | 41,727 | 417 | 9,532 / 32,195 | 3,496 / 3,531 / 34,700 |
| train_reduced_per_head | 33,118 | 417 | 17,067 / 16,051 | 6,992 / 7,062 / 19,064 |
| train_effective (after hflip) | 66,236 | 417 | 34,134 / 32,102 | 14,054 / 14,054 / 38,128 |
| validation_source | 7,022 | 73 | 1,706 / 5,316 | 658 / 722 / 5,642 |
| validation_reduced_per_head | 6,172 | 73 | 3,164 / 3,008 | 1,316 / 1,444 / 3,412 |
| test | 3,970 | 118 | 2,230 / 1,740 | 1,247 / 983 / 1,740 |

The validation clip set is recoverable from the artifact's
`background_reduction.validation.*.clips` lists (73 clips totalling exactly
7,022 frames), so the partition can be rebuilt without guessing a seed.

## Caveats for the paper

- The paper's Train/Val columns are `*_reduced_per_head`; its Test column is the
  raw natural split. Mixing the two in one table headed "extracted frames" is
  misleading — the natural pool is 41,727 / 7,022 / 3,970.
- What actually trains the model is `train_effective`, twice the Train column.
- Both heads' subtotals matching is a consequence of the per-head duplication,
  not of joint annotation.
- The paper's table reads 33,188 / 6,102 / 3,970 while run4 reads
  33,118 / 6,172 / 3,970: same policy and pool, a different clip draw (417/73
  instead of 392/98, i.e. a 0.15-ish validation fraction). No archived run
  matches the paper's exact triple.

## Related

- [[per-head-background-reduction]]
- [[sense-style-datasets]]
- [[deviation-label-join-collision]]
- [[kaggle-image-datasets]]
