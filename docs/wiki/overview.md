---
type: meta
tags: [overview, synthesis]
code_refs: [app.py, templates/index.html]
updated: 2026-06-06
---

# SenseCV - Overview

SenseCV is a single-user, local web tool for annotating walking clips that
combine phone video with IMU sensor data (accelerometer plus gyroscope). It
builds a clean labeled dataset from raw recordings: the operator finds the
usable portion of each clip, crops it, tags what happened, and exports a
trimmed video alongside time-aligned sensor data.

The working directory "Supermercado Telefrango (Sem GPS)" points at the
purpose: an indoor, GPS-free navigation/obstacle dataset captured while walking
a phone through a store. Velocity is estimated from IMU data rather than GPS;
see [[velocity-estimation]].

## How it fits together

```text
raw clip folders + exports/  -->  app.py (Flask)  -->  templates/index.html
2026_01_16-*/                       |                viewer + annotator
SenseCV roots/                      | estimates velocity,
exports/<folder>/                   | suggests crops,
  video.mp4 or <folder>.mp4         | serves videos
  frames/accel/rotations JSON       v
                              exports/<name>/ + history.json
                              trimmed MP4 + rebased sensor JSON
```

Exports are both outputs and selectable inputs. The viewer groups clips by
source folder and includes an `exports` group for reviewing exported clips.

## The pieces

| Layer | Page |
|---|---|
| Flask backend (data, suggestions, export) | [[app-backend]] |
| Browser viewer / annotator UI | [[viewer-frontend]] |
| Raw/export clip folder structure | [[clip-data-model]], [[exports-output]] |
| HTTP + JSON API surface | [[api-routes]] |
| GitHub project state | [[github-repository]] |
| Export ledger | [[history-json]] |
| Live DroNet inference | [[dronet-live-classification]] |
| Vendored DroNet runtime and research source | [[dronet-overview]], [[source-inventory]] |

## Core ideas

- [[velocity-estimation]] - dead-reckoning speed from IMU using a
  gyroscope-corrected orientation quaternion.
- [[orientation-detection]] - detecting when the phone is held vertical.
- [[walking-detection]] - detecting gait from acceleration-magnitude
  oscillation.
- [[crop-suggestion]] - auto-proposing crop windows; the UI tries lateral
  deviation first, then walking, then vertical.
- [[classification-taxonomy]] - the obstacle/response label scheme that names
  every export.
- [[export-pipeline]] - re-encoded video cuts and re-based sensor timestamps.
- [[video-sensor-sync]] - playing video with charts/cursor in lockstep.
- [[dronet-samsung-samples]] - random-frame DroNet inference over the Samsung
  30-05-2026 gimbal dataset.
- [[dronet-sensecv-02062026-3fps]] - fixed 3 FPS DroNet classifications and
  contact sheets over the 02-06-2026 gimbal dataset.
- [[dronet-live-classification]] - on-demand DroNet predictions in the Flask
  viewer for paused frames and 3 FPS playback.
- [[dronet-overview]] - the bundled DroNet source layer, including the
  PyTorch port, upstream Keras repo, paper sources, and offline inference
  scripts now kept under `dronet/`.

## End-to-end flow

1. The operator opens a clip. The backend computes per-frame accel, gyro, and
   estimated velocity; the UI draws charts.
2. The UI auto-suggests a crop, preferring lateral deviation windows and falling
   back to walking/vertical.
3. The operator verifies playback, adjusts start/end markers, and fills the
   classification.
4. Export re-encodes the MP4 cut, writes time-rebased sensor JSON, appends to
   [[history-json]], and refreshes the UI export state.

See [[index]] for the full catalog and [[log]] for the change timeline.


