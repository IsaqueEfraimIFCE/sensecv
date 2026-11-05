---
type: concept
tags: [training, sampling, two-head, gotcha]
code_refs: [scripts/mobilenetv1_two_head_all_frames.py, scripts/mobilenetv4_two_head_all_frames.py]
updated: 2026-08-16
---

# Per-head background reduction

Why the two-head all-frames trainers report more training rows than they have
frames, and what that means for a training sample.

## The mechanism

A single shared class balance cannot serve both heads, because `free_path` is
always `none`: reducing one head's background distorts the other's. So
`make_head_balanced_rows()` balances twice, independently:

- **obstacle head** — `free_path` frames are subsampled down to the obstacle
  count
- **deviation head** — `none` frames are subsampled down to left+right

Both reductions use the same largest-remainder allocation with evenly spaced
picks inside each clip. Each surviving row is then tagged:

```python
item["obstacle_active"], item["deviation_active"] = True, False   # obstacle copy
item["obstacle_active"], item["deviation_active"] = False, True   # deviation copy
```

A frame kept by both reductions therefore appears as **two rows**, and the
reported count (`combined_loss_samples`, `*_reduced_per_head`) is the sum.

## The labels do not change

Both copies carry both labels, identical. What differs is which head's loss the
row contributes to (`mobilenetv1_two_head_all_frames.py:432-434`):

```python
loss = ologits.sum() * 0.0
if obstacle_active.any():  loss += F.cross_entropy(ologits[obstacle_active],  obstacle[obstacle_active])
if deviation_active.any(): loss += F.cross_entropy(dlogits[deviation_active], deviation[deviation_active])
```

So a duplicated frame is forward-passed twice per epoch through the shared
backbone, training one head each time.

## Measured on run4's split

| | treino | validacao |
|---|---|---|
| rows | 33,118 | 6,172 |
| distinct frames | 24,347 | 4,288 |
| used by both heads | 8,771 | 1,884 |
| obstacle head only | 10,293 | 1,528 |
| deviation head only | 5,283 | 876 |

`8,771 x 2 + 10,293 + 5,283 = 33,118`.

Which frames duplicate is not arbitrary: 7,027 of the 8,771 are the
`obstacle`+`left`/`right` frames, foreground for **both** heads and therefore
never subsampled. The other 1,744 are background frames the two independent
draws happened to both keep.

## Consequences

- The backbone does not see the data uniformly: duplicated frames get twice the
  exposure per epoch, and four times once the hflip augmentation is applied
  (`train_effective` = 66,236).
- `train_loss` is a **sum** of two head terms, not a weighted mean, so it is not
  comparable to a single-head run's loss.
- Any dataset materialized "one image per frame" cannot reproduce the reported
  counts — see [[sense-style-datasets]].

## Superseded in later experiments

The 2026-08-16 MobileNetV4 runs ([[mobilenetv4-sense-experiments]]) drop this
scheme at the user's instruction: every frame carries both labels and trains
both heads, `loss = CE(obstacle) + CE(deviation)` with no masking and no
reduction. The datasets used there are already the materialized result of the
reduction, so the balance is preserved while the duplication is not.

## Related

- [[allframes-split-provenance]]
- [[sense-style-datasets]]
- [[mobilenetv4-sense-experiments]]
