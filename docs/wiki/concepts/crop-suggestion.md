---
type: concept
tags: [algorithm, workflow]
code_refs: [app.py, templates/index.html]
updated: 2026-06-11
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
| `lateral` | relative phone roll from accelerometer gravity direction | strongest sustained intentional left/right phone tilt over 1 s |

## Lateral deviation (`mode='lateral'`)
Current implementation: `suggest_lateral_deviation(idx)` reads `frames.json`
and `accelerations.json`, interpolates acceleration onto frame timestamps,
estimates phone roll with configurable `atan2(num_axis, den_axis)` axes
(defaults `y,z`), subtracts a neutral baseline from the first 0.5-1.0 s, and
smooths relative roll with a 100-200 ms rolling mean. It slides a 1 s window and
scores `mean_absolute_relative_roll * sign_consistency`, where
`sign_consistency = abs(mean(sign(relative_roll)))`.

Windows are rejected below `SENSECV_LATERAL_TILT_THRESHOLD_DEG` (default 0.0)
or below `SENSECV_LATERAL_SIGN_CONSISTENCY_THRESHOLD` (default 0.0). With both
defaults at zero, every clip returns its best-scoring window — full-scan
lateral detection is 589 of 589 clips. This restores the original
"must happen in every lateral deviation clip" behavior of the old
velocity-based detector (see below); the operator filters non-deviations by
not classifying those exports. Stricter gating is still available by setting
the env vars (the 2026-06-11 intermediate defaults 15.0 / 0.75 detected 68 of
589; the original 25.0 / 0.90 detected 45). The best
valid window returns start/end seconds, start/end frame IDs, score, mean roll,
mean absolute roll, max roll, sign consistency, and `predicted_direction`.
Positive mean roll is `right`; negative mean roll is `left`. Use
`SENSECV_LATERAL_ROLL_INVERT=1` if the phone axis is reversed.

The detector no longer uses long-term double integration of acceleration as the
main signal.

### Previous velocity notes
`suggest_lateral_deviation(idx)` in `app.py` returns the **1-second window
with the highest mean |lateral velocity|**, calibrated against the operator's
5 manually-cut deviation exports in [[history-json]] (entries 6-10, SenseCV
clips 28-31 and 46).

### Why lateral velocity
`_debug_lateral.py` benchmarks four candidate features against the 5 GT
windows. Result (mean of `|Δstart| + |Δend|` over the 5 clips, lower is
better):

| feature | mean offset |
|---|---|
| **lateral velocity, mean over 1 s window** | **0.61 s** ← current |
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
   trick as the gait detector — orientation-independent).
2. **Horizontal acceleration** = `linear − (linear · g_hat) g_hat`, projected
   onto an orthonormal 2-D basis in the horizontal plane.
3. **Velocity** = `cumsum(horiz × dt)` minus its clip mean (removes the IMU
   bias that `cumsum` would otherwise accumulate as drift).
4. **Local walking direction** = `dir_window_sec` (default 2 s) rolling mean
   of velocity. **Lateral velocity** = component perpendicular to it.
5. Slide a `window_sec`-wide window (default 1 s) and pick the position with
   the highest mean `|lateral velocity|`.

### Behavior
Always returns `found:true` with a window. Clips without a real deviation
still get one — intentional: the user requirement is "must happen in every
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

Available via `/api/suggest/<idx>?mode=lateral` and the "⬦ Desvio lateral"
button in [[viewer-frontend]]. Calibration scratchpad: `_debug_lateral.py`
in the project root — extend its `GT` list when more deviation cuts get
exported and re-run to compare candidate features.

## Horizontal (walking-only) clips
Clips from the extra input roots (the `SenseCV-*` folders, tracked in
`app.py` `WALKING_ONLY`) are filmed **horizontally the whole time**, so
orientation is meaningless for them. For these, `suggest_crop()` takes a
dedicated branch *before* the mode logic: it ignores the `vertical` mask
entirely and segments the first sustained **walking** run via
`_first_sustained(walking, …)` ([[walking-detection]]), still honoring a
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

## SSIM frame selection
Suggested cuts now run a visual de-duplication pass over the proposed video
window. The backend decodes frames from the source `video.mp4`, compares each
candidate against the last selected frame with SSIM, and keeps the frame when
similarity drops below `SENSECV_SSIM_THRESHOLD` (default `0.985`). The first and
last frames are always kept, and `SENSECV_SSIM_MAX_GAP_SEC` (default `0.5`)
forces periodic coverage through static stretches.

`/api/suggest/<idx>` includes `ssim.frames_before` and `ssim.frames_after` so
the UI can show how many source frames were in the cut and how many distinct
images remain for dataset composition.

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

