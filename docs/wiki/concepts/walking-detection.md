---
type: concept
tags: [algorithm, gait]
code_refs: [app.py]
updated: 2026-05-26
---

# Walking detection (gait)

The second per-frame mask from `_orientation_walking_masks()` in
[[app-backend]]: **`walking`**, true when the person is actually moving on foot.

## Criterion
Each footstep jolts the phone, so the **acceleration magnitude `|a|` oscillates**
during a walk and is nearly constant when standing still. The detector:
1. Computes `|a|` per frame.
2. Takes the rolling **standard deviation** of `|a|` over a 1-second window
   (mean via moving average, then `std = sqrt(mean((|a|-mean)^2)))`).
3. Flags `walking = std > 0.5 m/s²`.

Standing still gives std ≈ 0.05–0.2; walking pushes it well above 0.5, so the
threshold separates the two cleanly. This is a **heuristic**, not calibrated as
rigorously as [[orientation-detection]] — revisit the `0.5` constant if a clip
type misbehaves.

## Why magnitude std (not the fused velocity)
`|a|` is orientation-independent and needs no integration, so gait shows up
directly regardless of how the phone is tilted — robust and cheap. The fused
`speed` from [[velocity-estimation]] could be an alternative signal; comparing
them is a noted open question there.

## Role
Combined with the vertical mask in the **`walking` suggestion mode**: the crop is
the first sustained period that is *both* vertical *and* walking, trimming away
the standing/setup portion at the start. Observed effect: e.g. clip 0's window
tightens from `14.23→25.62` (vertical) to `22.39→25.62` (vertical+walking). See
[[crop-suggestion]].

