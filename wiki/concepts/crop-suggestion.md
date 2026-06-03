---
type: concept
tags: [algorithm, workflow]
code_refs: [app.py, templates/index.html]
updated: 2026-05-30
---

# Crop suggestion

`suggest_crop(idx, mode)` in [[app-backend]] proposes a `[start, end]` window for
a clip so the operator rarely sets markers by hand. Exposed at
`/api/suggest/<idx>?mode=` ([[api-routes]]).

## Three modes
| mode | mask used | intent |
|---|---|---|
| `vertical` | [[orientation-detection]] `vertical` | first sustained portrait period; the broader setup-to-end proposal |
| `walking` (auto) | exported history first; otherwise `vertical AND walking` ([[walking-detection]]) | walking-only crop that matches known exports and falls back to detector logic for new clips |
| `lateral` | gravity-removed horizontal accel magnitude | longest sustained lateral burst â‰¥ 0.2 s â€” for "swerve / sidestep" events |

## Lateral deviation (`mode='lateral'`)
`suggest_lateral_deviation(idx)` in `app.py` returns the **1-second window
with the highest mean |lateral velocity|**, calibrated against the operator's
5 manually-cut deviation exports in [[history-json]] (entries 6-10, SenseCV
clips 28-31 and 46).

### Why lateral velocity
`_debug_lateral.py` benchmarks four candidate features against the 5 GT
windows. Result (mean of `|Î”start| + |Î”end|` over the 5 clips, lower is
better):

| feature | mean offset |
|---|---|
| **lateral velocity, mean over 1 s window** | **0.61 s** â† current |
| z(lateral velocity) + z(yaw rate) combined | 0.53 s |
| yaw rate, mean over 1 s window | 0.68 s |
| peak \|lateral velocity\| in 1 s window | 1.48 s |
| rolling std of lateral velocity | 2.08 s |
| horizontal velocity magnitude | 7.11 s |
| lateral-accel magnitude (any variant) | matched only 1 of 5 GT |

The combined z-scored sum is marginally better but adds a second feature
without changing the algorithm shape; mean lateral velocity alone is simple
and within tolerance.

### Algorithm
1. **Gravity direction** = 1 s rolling mean of the raw accel vector (same
   trick as the gait detector â€” orientation-independent).
2. **Horizontal acceleration** = `linear âˆ’ (linear Â· g_hat) g_hat`, projected
   onto an orthonormal 2-D basis in the horizontal plane.
3. **Velocity** = `cumsum(horiz Ã— dt)` minus its clip mean (removes the IMU
   bias that `cumsum` would otherwise accumulate as drift).
4. **Local walking direction** = `dir_window_sec` (default 2 s) rolling mean
   of velocity. **Lateral velocity** = component perpendicular to it.
5. Slide a `window_sec`-wide window (default 1 s) and pick the position with
   the highest mean `|lateral velocity|`.

### Behavior
Always returns `found:true` with a window. Clips without a real deviation
still get one â€” intentional: the user requirement is "must happen in every
lateral deviation clip", and the operator filters non-deviations by simply
not classifying those exports.

### Calibration table (defaults, window_sec=1.0, dir_window_sec=2.0)
| clip | GT (operator) | detector | offset (s/e) |
|---|---|---|---|
| 179 | 5.01-5.95 | 4.49-5.48 | -0.52 / -0.47 |
| 180 | 4.68-5.48 | 4.53-5.51 | **-0.15 / +0.03** |
| 181 | 4.59-6.11 | 4.36-5.34 | -0.23 / -0.77 |
| 182 | 4.67-5.72 | 4.94-5.93 | +0.27 / +0.20 |
| 197 | 4.63-5.63 | 4.83-5.81 | +0.20 / +0.18 |

Mean abs offset 0.61 s, mixed signs (no consistent bias to correct).

Available via `/api/suggest/<idx>?mode=lateral` and the "â¬¦ Desvio lateral"
button in [[viewer-frontend]]. Calibration scratchpad: `_debug_lateral.py`
in the project root â€” extend its `GT` list when more deviation cuts get
exported and re-run to compare candidate features.

## Horizontal (walking-only) clips
Clips from the extra input roots (the `SenseCV-*` folders, tracked in
`app.py` `WALKING_ONLY`) are filmed **horizontally the whole time**, so
orientation is meaningless for them. For these, `suggest_crop()` takes a
dedicated branch *before* the mode logic: it ignores the `vertical` mask
entirely and segments the first sustained **walking** run via
`_first_sustained(walking, â€¦)` ([[walking-detection]]), still honoring a
`learned_walking_window()` if the clip was previously exported. The response
always reports `has_vertical:true` so the "no vertical moment" banner never
fires for footage that is intentionally horizontal. Only the supermarket footage
(the primary root) still uses vertical-orientation detection. See
[[orientation-detection]].

## Learned walking windows
`suggest_crop(idx, 'walking')` first checks `history.json` through
`app.py` `learned_walking_window()`. If a source clip has already been exported,
the latest exported `[start, end]` is returned exactly with `learned:true`.
Those exports are treated as hand-labeled walking-only ground truth.

For clips without export history, `walking` mode uses a supervised frame
classifier trained from the exported windows. `_classifier_feature_series()`
aligns accelerometer and gyroscope samples to video frames and builds features
for portrait state, acceleration energy, gyro energy, jerk, and rolling
step-cadence autocorrelation. `_walking_classifier_model()` trains a cached
Random Forest from `history.json`; `_classifier_walking_window()` smooths frame
probabilities and returns one contiguous walking segment.

## Window logic
`vertical` mode uses `_first_sustained()`:
- Scans for the first run of at least `1.5 s` of frames where the vertical mask
  holds, and uses that run's first frame as `start`.
- `end` is the last frame anywhere where the mask is true, so the window spans
  to the final qualifying portrait moment.

`walking` fallback uses `_classifier_walking_window()`:
- Classifies each frame as walking / not walking from IMU + gyroscope features.
- Uses rolling autocorrelation of acceleration magnitude to capture gait cadence.
- Smooths probabilities and merges nearby active pieces into one walking bout.
- Rejects short bursts and caps over-long detections using benchmark export
  durations.

## `has_vertical` and the warning
Every response includes `has_vertical = vertical.any()`. The UI shows the red
"no vertical moment" banner whenever this is false, independent of which mode
was asked for. If vertical exists but walking does not, `walking` mode returns a
different message ("sem momento de caminhada sustentado").

## UI integration ([[viewer-frontend]])
- Auto-suggest: `loadClip()` calls `runAutoSuggest()`, which tries `lateral`
  first, then `walking`, then `vertical`. The user preference is to start with
  the desvio-lateral cut and only use other selections as fallbacks.
- Three buttons ("Vertical", "Vertical + andando", "Desvio lateral") re-run on
  demand through one `runSuggest(mode)` with a stale-request guard.

## Tuning surface
The classifier is trained from the current exports in `history.json` and cached
until that file changes. Export history for the exact source clip takes
precedence over the classifier fallback.

