---
type: entity
tags: [backend, flask, python]
code_refs: [app.py]
updated: 2026-06-03
---

# app.py - Flask backend

The entire server in one file. Discovers and refreshes clips, serves video with
HTTP range support, computes per-frame sensor data plus estimated velocity,
suggests crops, and performs exports.

## Startup
- `CLIPS_DIR` = the project root; `EXPORTS_DIR` = `exports/`;
  `HISTORY_FILE` = `history.json`.
- Deployment can override persistence and mounted datasets with
  `SENSECV_EXPORTS_DIR`, `SENSECV_HISTORY_FILE`, and
  `SENSECV_EXTRA_CLIP_ROOTS`. Uploaded zip datasets are written to
  `SENSECV_UPLOADS_DIR`, defaulting locally to `uploaded_datasets/`.
- `find_clips()` scans `CLIPS_DIR`, `EXTRA_CLIP_ROOTS`, and `exports/` for
  clip-like subdirectories containing `frames.json` plus an `.mp4`. Raw clips
  usually use `video.mp4`; exports use `<folder>.mp4`, so `CLIP_VIDEO_PATHS`
  stores the exact video file per display name.
- Clips are addressed by integer index into `CLIPS`. `CLIP_PATHS` maps display
  names to folders; `CLIP_GROUPS` maps display names to the UI group selector.
- Template/page caching is disabled (`TEMPLATES_AUTO_RELOAD`, `no_cache()`) to
  avoid stale local UI after edits.
- Runs on `localhost:5000`, `threaded=True`.

## Multi-root input
Primary-root clips keep their bare folder name. Clips from each
`EXTRA_CLIP_ROOTS` entry are prefixed `label/name`, where `label` is the root's
basename. Exported clips are selectable as `exports/<folder>`.

As of 2026-06-02, `EXTRA_CLIP_ROOTS` includes the inner Samsung and Poco
30-05-2026 gimbal roots from Downloads plus the local
`SenseCV-02-06-2026-IFCE-Gimbal` root. The outer folders are wrappers; the app
points at the inner same-named directory that directly contains the clip ID
folders. Discovery verified 48 Samsung clips, 54 Poco clips, and 32 new
02-06-2026 gimbal clips, raising the app total to 313 selectable clips.

`WALKING_ONLY` holds the horizontal SenseCV roots and exports, so orientation
detection is skipped for those clips. See [[crop-suggestion]] and
[[orientation-detection]].

`refresh_clips()` rebuilds `CLIPS`, `CLIP_PATHS`, `CLIP_VIDEO_PATHS`,
`CLIP_GROUPS`, and `WALKING_ONLY`. It is called before rendering `/`, by
`/api/clips`, and by `/api/export-state`, so newly created or deleted export
folders can appear in the UI without restarting.

## Key functions

| Function | Role | Concept page |
|---|---|---|
| `interp_at_times()` | Linear interpolation of a sensor series onto frame timestamps | - |
| `compute_velocities()` | IMU dead-reckoning to per-frame `{vx,vy,vz,speed}` | [[velocity-estimation]] |
| `external_input_at_times()` | Aligns optional `external_sensors.json` button input to video frames | - |
| `get_clip_data()` | Assembles `fps/duration/times/accel/rotation/velocity/external_input`; LRU cache, max 5 clips | - |
| `_load_dronet_runtime()` | Lazily imports the Desktop DroNet PyTorch runtime and H5 weights | [[dronet-live-classification]] |
| `dronet_frame_classification()` | Decodes one video frame, runs DroNet, and returns steering/yaw/collision JSON | [[dronet-live-classification]] |
| `_orientation_walking_masks()` | Per-frame vertical plus walking boolean masks | [[orientation-detection]], [[walking-detection]] |
| `_first_sustained()` | First run >= 1.5 s where a mask holds | [[crop-suggestion]] |
| `suggest_crop()` | Crop proposal in `mode='vertical'` or `'walking'`; reports `has_vertical` | [[crop-suggestion]] |
| `suggest_lateral_deviation()` | Highest mean lateral-velocity window for sidestep/deviation clips | [[crop-suggestion]] |
| `_ffmpeg_cut()` | H.264 export cut preserving rotation metadata | [[export-pipeline]] |
| `save_sensor_data()` | Filters and re-bases sensor JSON to the crop window | [[export-pipeline]] |
| `save_ssim_review_videos()` | Writes SSIM audit videos for all/chosen/not-chosen crop frames | [[export-pipeline]] |
| `api_crop()` | ffmpeg cut plus sensor save plus history append | [[export-pipeline]] |
| `export_set()` | Batch driver; loops preset clips, runs ffmpeg and sensor save, writes `sources.csv` | [[SenseCV-dataset]] |
| `api_batch_export()` | `POST /api/batch-export` wrapper around `export_set()` | [[SenseCV-dataset]], [[api-routes]] |

## Caching note
`get_clip_data()` memoizes the heavy velocity computation per clip in an
`OrderedDict` capped at 5 entries. Re-opening a recent clip is instant; opening
a 6th evicts the least-recently used.

Live DroNet uses a separate cache. `_dronet_runtime` loads the model once on
first `/api/dronet/<idx>` request, and `_dronet_cache` stores up to 300
predictions keyed by clip display name, video mtime, and decoded frame index.
See [[dronet-live-classification]].

## fps derivation
`fps = 1e6 / median(diff(frame time_usec))`, robust to dropped frames versus a
naive count/duration.

## Routes
All HTTP endpoints are catalogued in [[api-routes]]. `/api/dronet/<idx>` is the
live neural inference endpoint used by the viewer's DroNet panel.
`/api/export-state` is the
frontend polling endpoint: it refreshes clip discovery, prunes `history.json`
against actual export folders, and returns clips/groups/history/export folders
plus the current next export number.

## Learned walking suggestions
`learned_walking_window()` reads `history.json` and returns the latest exported
`[start, end]` for a source clip. `suggest_crop(idx, 'walking')` uses that
window before running sensor heuristics, so exported walking-only clips become
ground truth for future suggestions. For clips with no export history,
`_classifier_feature_series()` computes IMU/gyroscope/gait-cadence features,
`_walking_classifier_model()` trains a cached Random Forest from exported
windows, and `_classifier_walking_window()` returns the single smoothed walking
bout.

## Launcher
`run_server.bat` and `run_server.ps1` set `PYTHONPATH` to the project folder,
stop existing listeners on port 5000, then run this checkout's `.venv` Python
against this checkout's `app.py`. This prevents accidentally serving an older
in-memory Flask process after code or template edits.

## Gotchas
- Batch export presets explicitly exclude display names starting with
  `exports/`; exported clips are selectable in the viewer but are not re-batched.
- Sensor-save failure inside `api_crop()` is swallowed as non-fatal; the video
  still exports even if sensor JSON cannot be written.
- `save_sensor_data()` preserves optional `external_sensors.json` for cropped
  exports, filtering to the crop window and re-basing `time_usec` like the IMU
  streams.


