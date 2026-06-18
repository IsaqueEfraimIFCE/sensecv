---
type: entity
tags: [frontend, html, chartjs]
code_refs: [templates/index.html]
updated: 2026-06-19
---

# templates/index.html - Viewer and annotator UI

A single self-contained page with inline CSS/JS and Chart.js from CDN. Server
Jinja injects bootstrap data: `CLIPS`, `CLIP_GROUPS`, `HISTORY`,
`EXPORT_FOLDERS`, and `NEXT_NUMBER`.

## Visual design
Dark theme driven by CSS custom properties in `:root`: a colour palette
(`--bg/--panel/--panel-2/--border/--text/--muted/--accent` + signal colours),
a **type scale** (`--fs-xs:13px … --fs-xl:26px`, base `--fs-base:16px`), radius
tokens (`--radius`/`--radius-sm`), and a `--shadow`. Body uses an Inter-first
system font stack with antialiasing and `line-height:1.5`. The whole UI was
**scaled up for readability**: the type scale was bumped (base 15→16px, etc.)
and the previously hard-coded `px` sizes throughout were moved onto the scale,
so almost all text now flows from `:root`. Headline readouts are large
(`.val-num` → `--fs-xl`, crop time → `--fs-lg`); the left panel widened to 540px
and the three charts grew to 190px tall to suit the bigger text. Modern touches:
pill tab bar with an accent-glow active state, rounded cards with soft shadows,
`:focus-visible` accent rings, and thin custom scrollbars.

## Layout
- **Header** - median fps / duration / frame-count badges, capture-quality
  badges, clip name, prev/next nav, a **folder/group selector**, a grouped clip
  selector, and a **tab bar** (`#tabbar`) selecting one of the six views.
- **Tabbed views** (see [[#Tabbed views]]) - the page is split into six
  client-side views; only the active one's panels show. The video + charts
  "player" stays mounted and is shown on the two video-centric views.
- **Persistent player** (`#player-left` + `.right`) - video, timeline canvas,
  time display with play button, vertical warning banner, chart toggles, live
  value boxes, and the three Chart.js charts (acceleration, gyroscope, estimated
  velocity, each with per-series toggles; acceleration also gets an `Input` trace
  when `/api/data` provides external button samples) plus the quick-actions strip.

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
  `lateral`, then `walking` (the retired `vertical` mode is no longer attempted).
  Lateral-deviation cuts are the first choice; walking is the fallback. A red
  banner appears if a vertical-dependent response reports no vertical moment.
  See [[crop-suggestion]].
- **Frame-selection visualizer** - lives in its **own `Frames` tab** (split out
  of `Anotar`), so the selected frames can be reviewed large. After an automatic
  or manual crop window, the panel loads `/api/ssim/<idx>` and renders the
  selected visually distinct frames. In the full-width `Frames` view the strip
  becomes a wrapping CSS grid (`repeat(auto-fill,minmax(340px,1fr))`) with images
  shown up to `62vh` tall at `object-fit:contain` — every retained frame is
  visible large and uncropped. A "Métrica" dropdown picks the similarity metric
  (SSIM, DINOv2, LPIPS, or VIF); changing it resets the threshold to that
  metric's default (0.985/0.98/0.95/0.70) and adjusts the slider range (VIF goes
  down to 0.5). Suggestions, thumbnail refreshes, and exports all send the chosen
  `metric` + `threshold`. Clicking a frame seeks the video to that retained
  source frame. `Anotar` keeps an **"Escolha de frames →"** button that switches
  to this tab; entering it re-runs `loadSsimVisualizer()` for the current crop.
- **Suggest buttons** - "⬦ Caminhada" (`walking`) and "⬦ Desvio lateral"
  (`lateral`), routed through `runSuggest(mode)` with a stale-request guard. The
  old "Vertical" button and `vertical` mode were retired (see [[crop-suggestion]]).
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

## Tabbed views
The UI is one page split into **six client-side views**, switched by the header
tab bar (no extra routes). `setView(name)` toggles a single `view-<name>` class on
`<body>`; CSS shows only `[data-view="<name>"]` panels for the active view. It
touches **only** `view-*` classes, never the `clip-vertical`/`clip-horizontal`
orientation classes. The last view is remembered in `localStorage`
(`sensecv-view`).

| Tab | `data-view` | Panels |
|---|---|---|
| Anotar | `anotar` | crop + classification + suggest buttons + IMU events + export |
| Frames | `frames` | métrica + threshold + large frame-selection grid + review links |
| Inferência | `inferencia` | DroNet + SenseCV `.keras` live panels + model swap |
| Lote + Manifesto | `lote` | batch export + manifest export |
| Desvios + Validação | `desvios` | deviation export + validation video |
| Datasets | `datasets` | `.zip` import + export history |

The persistent **player** (`#player-left` + `.right` charts) is shown only on
`anotar`/`inferencia`; the other four views hide it and span full width
(`.main` grid → single column). Because the panels stay in the DOM (hidden, not
removed), all existing element IDs and the DroNet/`.keras` live-update triggers
keep working across tabs — the frame-selection controls (`#ssim-metric`,
`#ssim-threshold`, …) work the same after moving from `Anotar` into `Frames`,
and `Anotar`'s Export still reads their values. On switching to a video view,
`setView()` calls `applyVideoOrientation()` and `chart.resize()` on the three
charts (Chart.js canvases compute zero size while their container is hidden); on
switching to `Frames` it calls `loadSsimVisualizer()`.

## Video orientation
`applyVideoOrientation()` decides rotation from the frame the browser **actually
paints**, and deliberately **ignores the container rotation tag**. It compares
the painted frame's `videoWidth/videoHeight` to the desired orientation
(`wantPortrait`, set by `loadClip()`: supermarket → portrait, else landscape)
and rotates 90° only when they disagree; a **square** or already-matching frame
is left untouched (`deg = 0`). When it does rotate, the element is sized to the
swapped box (wrapper height × width) so the rotated rectangle fills the wrapper.
It re-runs on `loadedmetadata` and `resize`.

> **Why not the tag?** An earlier version undid each clip's tag
> (`deg = (wantPortrait?90:0) - videoRotation`, from `_video_rotation()` /
> cv2 `CAP_PROP_ORIENTATION_META`). That broke the real footage: the IFCE gimbal
> set is 720×720 with a `rotate=90` tag **but its raw pixels are already
> upright**, and this browser does **not** honor the tag — so the "undo" spun
> upright frames onto their side (verified by decoding a raw frame). Measuring
> the painted frame is robust regardless of whether the browser honored the tag.
> `DATA.video_rotation` is still returned by the backend (`videoRotation` is kept
> in the payload) but is no longer used for orientation.

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
