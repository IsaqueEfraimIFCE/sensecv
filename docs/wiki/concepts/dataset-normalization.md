---
type: concept
tags: [dataset, normalization, sensors, video]
code_refs: [scripts/normalize_dataset_rates.py]
updated: 2026-06-12
---

# Dataset normalization

Historical note for a derived SenseCV dataset build named
`normalized_14_730fps_420hz`.

The normalization pass rebuilt clip folders so video frames use a fixed
14.730 FPS timeline and IMU/gyroscope streams use a fixed 420 Hz timeline. The
purpose was to make downstream frame selection, model training, and sensor/video
alignment operate on a consistent sampling grid instead of each phone capture's
native timing.

The wiki log records this as a 286-clip build produced by
`scripts/normalize_dataset_rates.py`. That script is not present in the current
checkout, so treat this page as documentation of the derived dataset contract,
not as a runnable workflow in this repository state.

## Contract

- Video timestamps are evenly spaced at 14.730 FPS.
- Accelerometer and gyroscope samples are resampled to 420 Hz.
- Clip folder shape remains compatible with [[clip-data-model]]:
  `frames.json`, `accelerations.json`, `rotations.json`, and the MP4 remain the
  source files consumed by [[app-backend]].
- Capture-quality badges in [[viewer-frontend]] and `/api/data` in
  [[api-routes]] can still report the observed minimum rates for the normalized
  streams.

## Related

[[video-sensor-sync]], [[velocity-estimation]], [[walking-detection]]
