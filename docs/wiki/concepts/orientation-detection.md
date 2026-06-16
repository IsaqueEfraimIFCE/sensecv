---
type: concept
tags: [algorithm, orientation]
code_refs: [app.py]
updated: 2026-05-27
---

# Orientation detection (vertical / portrait)

`_orientation_walking_masks()` in [[app-backend]] produces a per-frame boolean
**`vertical`** mask: true when the phone is held upright in portrait with the
camera pointing forward â€” the only orientation that yields usable footage.

## Criterion (per frame, then smoothed over ~1 s)
Using the nearest accelerometer sample, normalized by `|a|`:
- `ay/|a| > 0.95` â€” gravity lies almost entirely on the phone's **y-axis** â†’
  portrait (see axis convention in [[clip-data-model]]).
- `az/|a| < 0.12` â€” little gravity on **z** â†’ screen not facing up/down, i.e.
  camera roughly horizontal/forward.

Both signals are smoothed with a 1-second moving average (`W = fpsÂ·1`) before
thresholding, so brief wobbles don't toggle the mask.

## Calibration
Thresholds were tuned against 5 clips with known ground-truth start times
(13.0s, 11.5s, none, 9.0s, none) and reproduced **5/5** â€” see the docstring in
`_orientation_walking_masks()`. Treat these constants as calibrated; changing
them should be re-checked against those clips.

## Role
- The **default** crop suggestion uses this mask alone ([[crop-suggestion]]).
- `has_vertical = vertical.any()` â€” if a clip never goes vertical, the API flags
  it and the UI shows the **"no vertical moment"** warning ([[viewer-frontend]],
  [[api-routes]]).
- It is one of the two inputs to [[walking-detection]]'s combined mode.

## Applies only to the supermarket footage
This mask is meaningful **only for the primary-root supermarket clips**, which
are held vertical. Clips from the extra `SenseCV-*` roots (`WALKING_ONLY`) are
filmed horizontally throughout, so `suggest_crop()` skips the vertical mask for
them and segments walking alone, always reporting `has_vertical:true` to
suppress the warning. See [[crop-suggestion]] and [[app-backend]].

