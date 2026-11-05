---
type: concept
tags:
  - algorithm
  - workflow
code_refs:
  - app.py
  - templates/index.html
  - lllllllllllllllllll.ll.ll.
updated: 2026-07-28
"..........................................,.,,,. xxx/*v09*498-,v/-v8/,,,,,]4,vx6934lkx":
---
, mode)` in [[app-backend]] proposes a `[start, end]` window for
a clip so the operator rarely sets markers by hand. Exposed at
`/api/suggest/<idx>?mode=` ([[api-routes]]).

## Three modes
| mode | mask used | intent |
|---|---|---|
| `walking` (auto) | exported history first; otherwise `vertical AND walking` ([[walking-detection]]) | walking-only crop that matches known exports and falls back to detector logic for new clips |
| `lateral` | IMU deviation event (PDF, see below) | the deviation cut from `suggest_deviation_cut` — `decisao` window `[T1-Δ, T1]` by default |
| `trim` (2026-07-03) | IMU deviation cut, desvio search bounded | `suggest_trim_cut`: desvio (~1 s) searched only inside `[2s, duration-2s]` on clips ≥ 8 s; everything else = plain behavior (see below) |

> **Retired (2026-06-19):** the old `vertical` mode (first sustained portrait
> period, via `_first_sustained` on the orientation mask) was removed along with
> its viewer button. `suggest_crop(idx)` now only does walking; `mode` defaults
> to `'walking'` and `/api/suggest?mode=vertical` falls through to walking.
> `_first_sustained` is kept — it still segments the walking run for horizontal
> clips (below). The dead benchmark/template-matching detector
> (`_template_sustained` and its `_walking_templates`/`_benchmark_*`/
> `_walking_feature_series` helpers) and the orphaned
> `_lateral_acceleration_series` were deleted in the same pass.

> **As of 2026-06-18** the viewer's "Desvio lateral" button and
> `/api/suggest?mode=lateral` route to **`suggest_deviation_cut`** (the PDF cut,
> next section), not the roll/velocity heuristic below. The heuristic
> `suggest_lateral_deviation` now only backs the **batch** and **manifest**
> lateral exports (`export_set(mode='lateral')`, `export_manifest_clip`).

## Lateral deviation heuristic (`suggest_lateral_deviation`, batch/manifest only)
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

## PDF-based deviation cut (`suggest_deviation_cut`)
The viewer's "Desvio lateral" button (`/api/suggest?mode=lateral`) **and** the
deviation export/validation feature ([[api-routes]] `/api/inspect-deviation`,
`export_set(mode='deviation')`) both use `suggest_deviation_cut(idx)`, which
follows the criterios PDF
(`criterios_imu_video_orientando`, see [[imu-event-labeling]]): it takes the
strongest `desvio` event from `detect_imu_events` — physical start **T1**, return
to a stabilized trajectory **T2** — and returns one of the PDF §2.2 label
windows:

| window | range | intent |
|---|---|---|
| `decisao` (default) | `[T1-Δ, T1]` | the scene **before** the body reacts; PDF's recommended target for a predictive CNN (§2.1/§10 "regra de ouro") |
| `acao` | `[T1, T2]` | the deviation execution |
| `expandido` | `[T1-Δ, T2+margem]` | both, robust to sync uncertainty |

`Δ` = `SENSECV_IMU_EVENT_DELTA_SEC` (default 1.0 s), `margem` =
`SENSECV_IMU_EXPANDED_MARGIN_SEC` (0.3 s); the window is chosen with
`SENSECV_DEVIATION_CUT_WINDOW` (default `decisao`). The side (`LEFT`/`RIGHT`)
is the event's own `direction`, so cut and label are always consistent. The
desvio response carries `event_type: 'desvio'`. This is the cut written by
`export_set(mode='deviation')`.

### Stop-onset fallback (no desvio — cadence loss near the clip end)
When a clip has **no `desvio`**, `suggest_deviation_cut` falls back to **the
moment the step rhythm disappears**. `detect_stop_onset(idx)` computes sliding
gait periodicity with `_step_rhythm()`. It requires periodic stepping earlier
in the clip, rejects clips still stepping in the final analysis window, and
places **T1** at the first cadence window after the last detected stepping
window. T1 must fall in the final `SENSECV_IMU_STOP_TAIL_SEC` seconds. The same
PDF label window is cut around it (`decisao` → `[T1-Δ, T1]`), with
`event_type:'parada'`, `stop_time:T1`, and `side:'NONE'`.

`SENSECV_IMU_STOP_ONSET_MIN_SEC` and `SENSECV_IMU_STOP_CONFIRM_SEC` remain
declared constants but are not referenced by the current
`detect_stop_onset()`. A historical scan recorded 0 parada cuts in 389 clips;
rerun that audit after detector or dataset changes.

### Free-walk fallback (no desvio, no parada — detectable gait)
A clip with **neither a desvio nor a parada** can be a **free walk** (caminhada
livre), a legitimate third dataset class alongside desvio and parada. It
qualifies when `_step_rhythm()` contains at least one window whose
autocorrelation periodicity reaches `IMU_STEP_PERIODICITY`. The current
`_livre_cut()` returns the **whole video** `[0, duration]`, with
`event_type:'livre'`, `side:'NONE'`, and `direction:None`. A clip with no
discernible periodic gait returns `found:false`
(`'nenhum desvio, parada ou caminhada detectados'`).

Current scan (389 clips): **162 desvio, 0 parada, 97 livre, 130 none** (the 130
fall through to `walking` mode in the viewer).

The export/validation surfaces inherit this through the shared return shape:
`export_set(mode='deviation')` exports the free-walk cut like any other, but
`export_deviation_set` still keeps **only** `side ∈ {LEFT, RIGHT}`, so the
deviation validation video stays desvio-only — `livre` and `parada` (both
`side: 'NONE'`) are filtered out of it.

## Margin-bounded cut (`suggest_trim_cut`, mode `trim`)
Added 2026-07-03: same logic as `suggest_deviation_cut`, with one added rule
for desvio. The rationale is the recording convention: the operator is
starting/stopping the capture in the first and last moments, so a desvio cut
never sits in those edges ("miolo").

- **desvio**: the ~1 s window is only *looked for* inside
  `[margin, duration - margin]` (default 2 s each side). An event whose cut
  window falls outside is **discarded, never shifted**; the clip then falls
  through the normal fallback chain. The manifest signal-recovery search
  (`_desvio_window_from_signal`) is bounded the same way — its new
  `t1_min`/`t1_max` params mask the peak search and stop the onset walk, so a
  recovered T1 genuinely lands inside the margins (real in-margin timings, not
  boundary-clamped clones).
- **Short clips are exempt**: any clip under `TRIM_CUT_MIN_DURATION_SEC`
  (default **8 s**, env-overridable) skips the margins entirely and returns
  exactly the plain deviation cut (full-clip search).
- **parada and livre are untouched** in every case: parada keeps its 1 s
  stop-onset window with the whole clip searched; livre keeps the whole clip.

Implementation: `suggest_trim_cut(idx, margin=None)` is a thin wrapper over
`suggest_deviation_cut(idx, cut_margin=margin)` (the new `cut_margin` param
carries all the behavior above; it self-disables for short clips); the wrapper
retags `mode:'trim'` and adds `trimmed:true` + `margin` to the response, which
otherwise keeps the deviation cut's own payload (`event_type`, `side`,
`t1/t2`, …). The margin is `SENSECV_TRIM_CUT_MARGIN_SEC` (default **2.0 s**)
per side. Exposed at `/api/suggest/<idx>?mode=trim` ([[api-routes]]) with the
same SSIM frame-selection wrapper as the other modes, and in
[[viewer-frontend]] as the "⬦ Miolo 2s" button; the cut-type chip shows the
resolved class with a `· miolo 2s` suffix. Since 2026-07-03 it is also the
**default on-load suggestion**: `runAutoSuggest()` tries `trim` then
`walking`; the "⬦ Desvio lateral" button remains the unbounded cut.

Verified sample (398-clip root, first 150): the 30 clips under 8 s return
windows identical to the plain cut; the parada clip keeps `17.967–18.967`;
long-clip desvios show 0 margin violations (2 edge desvios re-detected
in-margin).

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
Horizontal (walking-only) clips use `_first_sustained()` on the walking mask:
- Scans for the first run of at least `1.5 s` of frames where the mask holds,
  and uses that run's first frame as `start`.
- `end` is the last frame anywhere where the mask is true, so the window spans
  to the final qualifying moment.

`walking` mode for supermarket clips uses `_classifier_walking_window()`:
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
- Auto-suggest: `loadClip()` calls `runAutoSuggest()`, which tries `trim`
  first (2026-07-03; previously `lateral`), then `walking`. The user preference
  is to start with the deviation cut *with the miolo margins applied* and use
  walking only as a fallback. Since `trim` keeps the PDF cut's
  desvio → parada → **livre** fallback chain, it returns `found:true` for any
  clip that contains walking, so auto-suggest almost always settles on `trim`;
  it only falls through to `walking` when the clip has no usable walking at
  all. The toast tags the result `caminhada livre` when `event_type==='livre'`.
- Three buttons ("⬦ Caminhada", "⬦ Desvio lateral", "⬦ Miolo 2s") re-run on
  demand through one `runSuggest(mode)` with a stale-request guard, under the
  **Anotar** tab.
- **Cut-type badge** (`#cut-type-disp`): after every suggest cut (auto or
  button), a persistent "Tipo de corte:" chip under the suggest buttons shows
  the resolved class — `Desvio lateral (esquerda/direita)`, `Parada @ <t>`,
  `Caminhada livre`, or `Caminhada` — colour-coded by `data-type`
  (desvio/parada/livre/caminhada). `cutTypeInfo(d, mode)` derives it from the
  response `event_type`/`side`; `hideCutType()` clears it on clip change.

## Tuning surface
The classifier is trained from the current exports in `history.json` and cached
until that file changes. Export history for the exact source clip takes
precedence over the classifier fallback.
