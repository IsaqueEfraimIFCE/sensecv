---
type: entity
tags: [frontend, html, chartjs]
code_refs: [templates/index.html]
updated: 2026-06-03
---

# templates/index.html - Viewer and annotator UI

A single self-contained page with inline CSS/JS and Chart.js from CDN. Server
Jinja injects bootstrap data: `CLIPS`, `CLIP_GROUPS`, `HISTORY`,
`EXPORT_FOLDERS`, and `NEXT_NUMBER`.

## Layout
- **Header** - fps / duration / frame-count badges, clip name, prev/next nav,
  a **folder/group selector**, and a grouped clip selector. The group selector
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
- **Auto-suggested crop** - on `loadClip()`, `runAutoSuggest()` tries
  `lateral`, then `walking`, then `vertical`. Lateral-deviation cuts are the
  first choice; other modes are fallbacks only. A red banner appears if a
  vertical-dependent response reports no vertical moment. See [[crop-suggestion]].
- **SSIM frame visualizer** - after an automatic or manual crop window, the
  SSIM panel can load `/api/ssim/<idx>` and render the selected visually
  distinct frames as clickable thumbnails. The operator can adjust the SSIM
  threshold in the crop panel; suggestions, thumbnail refreshes, and exports all
  use that value. Clicking a thumbnail seeks the video to that retained source
  frame.
- **Suggest buttons** - "Vertical", "Vertical + andando", and "Desvio lateral",
  all routed through `runSuggest(mode)` with a stale-request guard.
- **Export state polling** - `refreshExportState()` calls `/api/export-state`
  every 5 seconds. It refreshes `CLIPS`, `CLIP_GROUPS`, `HISTORY`,
  `EXPORT_FOLDERS`, and `NEXT_NUMBER`, so deleted/new export folders and pruned
  history appear without a manual restart.
- **Batch export panel** - Supermercado/SenseCV x walking/lateral. Each button
  POSTs to `/api/batch-export` and reports ok/skipped/failed counts in a toast.
  Batch presets intentionally exclude `exports/` clips.
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
`/api/data/<idx>`, `/api/dronet/<idx>`, `/api/suggest/<idx>`,
`/api/ssim/<idx>`, `/api/crop`, `/api/export-state`, `/api/history`).

