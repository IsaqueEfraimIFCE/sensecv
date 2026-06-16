---
type: entity
tags: [frontend, html, chartjs]
code_refs: [templates/index.html]
updated: 2026-06-16
---

# templates/index.html - Viewer and annotator UI

A single self-contained page with inline CSS/JS and Chart.js from CDN. Server
Jinja injects bootstrap data: `CLIPS`, `CLIP_GROUPS`, `HISTORY`,
`EXPORT_FOLDERS`, and `NEXT_NUMBER`.

## Layout
- **Header** - median fps / duration / frame-count badges, minimum observed
  video FPS / IMU Hz / gyro Hz capture-quality badges, clip name, prev/next
  nav, a **folder/group selector**, and a grouped clip selector. The group selector
  jumps directly to the first clip in a source group such as
  `Supermercado Telefrango`, a SenseCV folder, or `exports`.
- **Left column** - video, timeline canvas, time display with play button,
  vertical warning banner, chart toggles, live value boxes, the DroNet panel,
  the crop and classification panel, batch export controls, and export history.
- **Right column** - three Chart.js line charts: acceleration, gyroscope, and
  estimated velocity. Each has per-series toggles; the acceleration chart also
  has an `Input` trace when `/api/data` provides external button samples.

## Clip navigation
`buildClipSelect()` renders the clip dropdown with `<optgroup>` sections from
`CLIP_GROUPS`. `buildGroupSelect()` renders a separate folder dropdown using the
first index for each group. Selecting a folder calls `loadClip()` for that
group's first clip, avoiding a long scroll through every file. Selecting a clip,
using prev/next, or opening history keeps the group selector synced.

Export folders are selectable as `exports/<folder>`. The backend serves their
actual `<folder>.mp4` file rather than assuming `video.mp4`.

## Behaviors
- **Playback synced with sensors** - play/pause button plus spacebar; a
  `requestAnimationFrame` loop moves the chart cursor, value boxes, and timeline
  playhead smoothly while playing. See [[video-sensor-sync]].
- **External input overlay** - `DATA.external_input[]` is aligned to frame
  times by the backend. Samples equal to `1` appear as an orange stepped trace
  at the top of the acceleration chart.
- **Timeline canvas** - draws the speed waveform, crop region, S/E markers, and
  playhead; click/drag to scrub.
- **DroNet live panel** - Shows direction/steering, yaw, and collision
  probability from `/api/dronet/<idx>`. Playback requests are deduped to 3 FPS;
  paused and scrubbed states request the exact paused frame. See
  [[dronet-live-classification]].
- **SenseCV (.keras) panel** - Shows the local two-head model's obstacle and
  deviation prediction (label + probability) for the current frame via
  `/api/sensemodel/<idx>`. Driven from inside `requestDronet()`, so it shares
  the DroNet triggers (3 FPS while playing, exact frame when paused/scrubbed)
  and reset lifecycle. Degrades to an error line if TensorFlow or the
  `best_model.keras` file is absent.
- **Auto-suggested crop** - on `loadClip()`, `runAutoSuggest()` tries
  `lateral`, then `walking`, then `vertical`. Lateral-deviation cuts are the
  first choice; other modes are fallbacks only. A red banner appears if a
  vertical-dependent response reports no vertical moment. See [[crop-suggestion]].
- **Frame-selection visualizer** - after an automatic or manual crop window,
  the panel loads `/api/ssim/<idx>` and renders the selected visually distinct
  frames as clickable thumbnails. A "Métrica" dropdown picks the similarity
  metric (SSIM, DINOv2, LPIPS, or VIF); changing it resets the threshold to
  that metric's default (0.985/0.98/0.95/0.70) and adjusts the slider range
  (VIF goes down to 0.5). Suggestions, thumbnail refreshes, and exports all
  send the chosen `metric` + `threshold`. Clicking a thumbnail seeks the video
  to that retained source frame.
- **Suggest buttons** - "Vertical", "Vertical + andando", and "Desvio lateral",
  all routed through `runSuggest(mode)` with a stale-request guard.
- **Eventos IMU panel** - a Δ selector (0.5/1.0/1.5 s) and "Detectar" button
  call `/api/imu-events/<idx>` and list the capture verdict plus each detected
  desvio/redução/parada with T1/T2, direction, and confidence. Each event has
  three window buttons (Decisão, Ação, Expandido) that apply the range as the
  crop window and refresh the frame visualizer. See [[imu-event-labeling]].
- **Export state polling** - `refreshExportState()` calls `/api/export-state`
  every 5 seconds. It refreshes `CLIPS`, `CLIP_GROUPS`, `HISTORY`,
  `EXPORT_FOLDERS`, and `NEXT_NUMBER`, so deleted/new export folders and pruned
  history appear without a manual restart.
- **Batch export panel** - POSTs `/api/batch-export` for `all`, `supermarket`,
  or `sensecv` presets in `walking` or `lateral` mode. The response links the
  generated `sources.csv` and `review_all_frames.mp4`, a single video with all
  frames from the exported cuts for visual review.
- **Manifest export panel** - two mode buttons (`walking` / `lateral`). On
  click, `runManifestExport()` POSTs `/api/manifest-export` to plan the queue,
  then POSTs one clip at a time to `/api/manifest-export/clip` (updating an
  `exported / total` progress line and stopping on the exact failing
  `source_display`), and finally POSTs `/api/manifest-export/review` to link the
  combined review MP4. An empty queue (no `.xlsx` manifest) reports so without
  error.
- **Dataset import** - A zip upload control POSTs `.zip` archives to
  `/api/upload-zip`. The backend extracts nested SenseCV-style clip folders,
  refreshes discovery, and the UI rebuilds the selectors from the returned
  clip/group lists.
- **Classification chips** - occurrence/position/response/direction; chosen
  values plus name compose the export folder name live. Collision checks use
  actual export folders from `EXPORT_FOLDERS`.
- **Export** - POSTs to `/api/crop`, then refreshes export state.

## Keyboard shortcuts
`Space` play/pause; left/right seek 0.5 s (`Shift` = 5 s); `s` mark start;
`e` mark end; `[` / `]` previous / next clip.

## Talks to
[[api-routes]] (`/video/<idx>`, `/frame/<idx>/<source_index>.jpg`,
`/api/data/<idx>`, `/api/dronet/<idx>`, `/api/sensemodel/<idx>`,
`/api/suggest/<idx>`, `/api/ssim/<idx>`, `/api/imu-events/<idx>`, `/api/crop`,
`/api/batch-export`, `/api/manifest-export`, `/api/manifest-export/clip`,
`/api/manifest-export/review`, `/api/export-state`, `/api/history`,
`/api/upload-zip`, `/derived-file/<path>`).
