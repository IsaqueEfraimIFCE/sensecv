# Wiki Log

Append-only. One entry per operation. Greppable prefix:
`grep "^## \[" log.md | tail -5`.

---

## [2026-06-18] change | viewer "Desvio lateral" button uses the PDF cut
- `/api/suggest?mode=lateral` now routes to `suggest_deviation_cut` (PDF decisão
  window T1-Δ→T1), matching the deviation export. Clips with no IMU desvio
  return `found:false`, so `runAutoSuggest` falls back to walking/vertical.
  `suggest_lateral_deviation` now backs only the batch/manifest lateral exports.
- Verified: `/api/suggest/0?mode=lateral` → found, [1.471, 2.471], window
  decisao, side LEFT, SSIM enrichment intact. Frontend unchanged (it reads
  start/end/ssim/has_vertical/message, not mode).
- Pages touched: [[crop-suggestion]], [[api-routes]].

## [2026-06-18] feat | deviation cut follows the criterios PDF (T1-Δ→T1)
- Added `suggest_deviation_cut(idx)`: takes the strongest `desvio` from
  `detect_imu_events` and returns a PDF §2.2 label window around T1/T2. Default
  `SENSECV_DEVIATION_CUT_WINDOW=decisao` → `[T1-Δ, T1]`, the "decisão visual"
  scene before the body reacts (PDF's recommended target for a predictive CNN;
  user-chosen). `acao` / `expandido` also available. Side comes from the same
  event so cut and label stay consistent.
- `export_set` gained `mode='deviation'`; the deviation export
  (`/api/inspect-deviation`) now cuts with this PDF window instead of
  `suggest_lateral_deviation`, and drops the old `_clip_deviation_side` helper.
- Verified: clip 01 t1=2.471 → cut [1.471, 2.471]; 6 clips exported with the
  full common-export folder content (mp4 + frames/accel/rotation json + ssim),
  validation.mp4 30 fps / 174 frames.
- Pages touched: [[crop-suggestion]], [[imu-event-labeling]], [[api-routes]].

## [2026-06-18] fix+rework | inspection video uses suggested cut + sensor side
- Reworked `/api/inspect-deviation` to drop the ML model entirely. Each clip now
  contributes **exactly its `suggest_lateral_deviation` cut**, labelled with the
  **sensor** deviation side from `detect_imu_events` (`_clip_deviation_side`:
  desvio direction esquerda/direita→LEFT/RIGHT, none→NONE). Title card + banner
  per cut; output cadence `play_fps` (default `SENSECV_INSPECT_FPS=15`).
- Fixed two latent bugs surfaced by the rework:
  - **`KeyError: '<clip name>'`** during long runs. `find_clips()` did
    `CLIP_PATHS.clear()` + repopulate; with `threaded=True`, the 5 s
    `/api/export-state` poll cleared the shared maps mid-rebuild while the
    inspection loop indexed them. Now `find_clips()` builds local dicts and
    **swaps the globals atomically** — readers see old or new, never empty.
  - **`_centered_rolling_mean` was undefined** (lost in restoration), so
    `detect_imu_events` / `/api/imu-events` raised `NameError`. Defined it as a
    list-tolerant centered moving average next to `_moving_average`.
- Verified (venv, TF present but unused): count=8 → 384 frames, 15 fps, 720×720;
  sides LEFT×3 RIGHT×5; `/api/imu-events/0` and `/api/sensemodel/0` still 200.
- Pages touched: [[api-routes]], [[imu-event-labeling]].

## [2026-06-18] feat | deviation inspection video (N clips or all)
- New `POST /api/inspect-deviation` `{count: N|"all", sample_fps?}` builds one
  stitched MP4 over the chosen clips with the active `.keras` model. Each clip
  is preceded by a title card naming its **dominant deviation**
  (`LEFT/RIGHT/NONE`, majority vote via `_dominant_label`) and every sampled
  frame carries the live `Desvio`/`Obstaculo` overlay (`_draw_label_bar`).
  Output `data/derived/deviation_inspection.mp4`; no `exports/`/history writes.
- Refactored the shared inference path into `_sensemodel_preprocess()` +
  `_split_head_arrays()` (reused by `sensemodel_frame_classification` and the
  new batched builder). Added `INSPECT_SAMPLE_FPS`/`SENSECV_INSPECT_FPS`.
- `templates/index.html`: new "Vídeo de inspeção (desvio)" panel with an
  "Inspecionar N" count input and "Todos" button (`runInspect`).
- Verified with the venv Python (TF 2.17): count=2 → 720×720, 4 fps, 69 frames,
  clip 01 NONE, clip 02 LEFT+OBSTACLE; output MP4 decodes in OpenCV.
- Pages touched: [[api-routes]].

## [2026-06-18] feat | upload any .keras model at runtime
- Made the SenseCV live model swappable without restarting Flask. The active
  path now lives in `_sensemodel_runtime['path']` (default still
  `SENSECV_MODEL_PATH` → `best_model.keras`); `_set_sensemodel_path()` resets
  the cached model and the per-frame prediction cache.
- New routes: `POST /api/upload-model` (saves to `data/models/`, activates and
  eagerly loads the upload), `POST /api/reset-model`, `GET /api/sensemodel-info`.
  Added `SENSECV_MODELS_DIR` (default `data/models`). `/api/sensemodel` now
  echoes the active `model` name.
- `templates/index.html`: SenseCV panel shows "Modelo ativo: <name>" and a
  **Trocar modelo / Padrão** upload row; `swapModel()` posts the file, relabels
  the panel, and re-runs inference on the current frame.
- Verified via Flask test client: info reports the default; a bogus upload saves
  with a sanitized name and returns an actionable load error (400); reset
  restores `best_model.keras`.
- Pages touched: [[api-routes]], [[dronet-live-classification]].

## [2026-06-16] fix | sensemodel preprocessing (raw 0-255, not /255)
- `/api/sensemodel` returned near-constant predictions. Cause: the exported
  model is MobileNetV2 with an embedded `Rescaling(1/127.5, -1)` layer, so it
  expects raw 0-255 input; the route was also dividing by 255, collapsing the
  effective range to ~[-1,-0.992]. Removed the `/255.0` — predictions now vary
  per frame (obstacle flips NONE/OBSTACLE, deviation LEFT/RIGHT/NONE). Color
  stays BGR→RGB, 224×224×3. Documented in [[api-routes]].

## [2026-06-16] frontend | wired SenseCV + manifest panels, enabled .keras model
- Added the **SenseCV (.keras) live panel** and **manifest export panel** to
  `templates/index.html`, the two UIs documented in [[viewer-frontend]] but
  missing from the recovered frontend. SenseCV inference piggybacks on
  `requestDronet()` so it shares the play/pause/scrub triggers; manifest export
  plans → exports clip-by-clip → builds the review MP4.
- Installed `tensorflow-cpu` into `.venv` and copied the user's
  `best_model.keras` (224×224×3, two heads 2+3) to the project root, so
  `/api/sensemodel` now returns real predictions instead of the controlled 503.
  Added `requirements-sensemodel.txt`.
- Deleted the stale `C:\Users\Isaque\Desktop\senseCV` copy (was junk + a subset
  of this checkout); its lone unique file was the lost-panel spec, now realized.

## [2026-06-15] recovery | rebuilt deleted wiki + re-implemented lost routes
- The `docs/wiki` tree was deleted; restored from the GitHub base plus replayed
  session edits, and the 11 DroNet/meta pages were copied from the intact
  `C:\Users\Isaque\sensecv` checkout. Both wiki copies are back to 38 pages.
- Recovered two dataset scripts missing from `sensecv` entirely
  (`make_clear_ssim_last1s_dataset.py`, `make_clear_metric_last1s_dataset.py`)
  from transcripts into `scripts/`.
- `/api/manifest-export` (+`/clip`, `/review`) and `/api/sensemodel` had no
  surviving source anywhere (only this wiki documented them), so they were
  **re-implemented from the spec**, not recovered: `plan_manifest_export()`,
  `export_manifest_clip()`, `build_manifest_review()`, and
  `sensemodel_frame_classification()` in `app.py`, mirroring the DroNet
  lazy-load + graceful-degrade pattern. Added `openpyxl` to `requirements.txt`.
- Verified by Flask test client: planning with a synthetic `.xlsx` queued 3
  clips with composed labels, a clip export cut + sensor/SSIM save succeeded,
  and the review MP4 built; `sensemodel` returns its controlled 503 (no `.keras`
  model in this checkout). Frontend wiring (batch manifest panel, sensemodel
  panel) is NOT yet added — backend + docs only.
- pages touched: [[api-routes]], [[log]]

## [2026-06-12] feature | IMU event labeling from the criterios PDF
- Implemented `data/uploaded_datasets/criterios_imu_video_orientando.pages.pdf`:
  `imu_quality_report()` (section 1 checklist + verdict), `detect_imu_events()`
  (desvio/reducao/parada with T1/T2, acao/decisao/expandido windows, and
  alta/baixa/descartar confidence), route `/api/imu-events/<idx>?delta=`,
  viewer "Eventos IMU" panel, and `scripts/imu_event_report.py` batch CSVs.
- Desvio uses gravity-projected yaw rate when the capture rotates, falling
  back to drift-corrected lateral velocity on gimbal-stabilized clips
  (max yaw ≈ 11°/s there). Validated on the 32 IFCE manifest clips: 26/32
  correct directions, 3 misses, 2 of 3 wrong flagged baixa.
- Pages touched: [[imu-event-labeling]] (new), [[api-routes]],
  [[viewer-frontend]], [[index]], [[log]].

## [2026-06-12] deploy | metric selector live locally and on Fly
- Local: `scripts/run_server.bat` on port 5000; all four metrics verified
  (`lpips`/`sewar` installed into `.venv`).
- Fly: redeployed `sensecv-api` (machine version 6). SSIM + VIF work; DINOv2 +
  LPIPS return a controlled "indisponivel" error (no torch in the slim image).
  Added `sewar` to `requirements.txt`.
- Fixed two deploy bugs: `[build].dockerfile` path in `deploy/fly.toml` is
  config-relative (`"Dockerfile"`), and `find_clips()` skipped the recursive
  scan when `CLIPS_DIR` == uploads root (`/data/clips`), which made the Fly app
  report `clips=0`.
- Pages touched: [[deployment-operations]], [[log]].

## [2026-06-12] feature | metric selector for frame selection in the viewer
- `app.py`: `ssim_frame_selection()` now takes `metric='ssim'|'dinov2'|'lpips'|'vif'`
  with lazy-loaded backends and per-metric default thresholds
  (`DEFAULT_METRIC_THRESHOLDS`); `/api/ssim` + `/api/suggest` accept `?metric=`,
  `/api/crop` accepts body `ssim_metric`. Score still reported as `ssim_prev`.
- `templates/index.html`: "Métrica" dropdown beside the threshold control;
  changing it resets the threshold to the metric default and widens the slider
  range for VIF (min 0.5). Suggest, visualizer, and export all send the metric.
- Verified via Flask test client: all 4 metrics return selections on clip 0;
  invalid metric → 400.
- Pages touched: [[api-routes]], [[viewer-frontend]], [[log]].

## [2026-06-12] ingest | DINOv2/LPIPS/VIF clear last-1s dataset variants
- Added `scripts/make_clear_metric_last1s_dataset.py`: same last-1-second
  selection pipeline as the SSIM clear supplement, with a pluggable metric
  (`--metric dinov2|lpips|vif`).
- Built all three over the same 112 uploaded clips (112/112 ok, labels match
  files): DINOv2 0.98 → 444 images, LPIPS 0.95 → 476, VIF 0.70 → 1386
  (SSIM 0.95 reference: 517). VIF's default threshold was lowered to 0.70
  because consecutive-frame VIF sits at 0.60–0.93.
- New deps installed: `lpips` 0.1.4, `sewar` 0.4.8 (DINOv2 via torch.hub).
- Variants are not merged into `data/derived/clear/images/`.
- Pages touched: [[clear-dataset]], [[index]].

## [2026-06-11] docs | synchronized wiki after clear repair and normalization
- Added [[clear-dataset]] to document the clear/no-obstacle image pool, the
  517-image SSIM 0.95 supplement, and the 12 stale label rows removed from
  `clear_ssim095_last1s_until_112`.
- Added [[dataset-normalization]] for the historical
  `normalized_14_730fps_420hz` build: 14.730 FPS video and 420 Hz sensor
  streams.
- Updated backend/API/frontend wiki pages for current `data/` paths, recursive
  dataset discovery, capture-quality badges, `/api/sensemodel`, and local
  model/runtime notes.
- Pages touched: [[clear-dataset]], [[dataset-normalization]], [[api-routes]],
  [[viewer-frontend]], [[app-backend]], [[index]], [[log]].

## [2026-06-09] git | published source-only project commit to GitHub
- Read the wiki and followed the [[github-repository]] source-only policy:
  local datasets, exports, `history.json`, `.env`, logs, generated media, and
  Python caches remain ignored.
- Added `.env` to `.gitignore` while keeping `.env.example` tracked.
- Created local commit `99dd97a Commit SenseCV project source`, then found the
  checkout had no `origin` remote configured, so the commit was not visible on
  GitHub.
- Added `origin https://github.com/IsaqueEfraimIFCE/sensecv.git`.
- Initial push was rejected because GitHub already had remote history through
  `bcf1859 Load Fly example dataset`; histories were unrelated.
- Preserved the local root commit as branch `local-root-snapshot`, created a
  publish commit on top of `origin/master`, and pushed a fast-forward update:
  `bcf1859..8d3c6ad`.
- Current GitHub `master` is
  `8d3c6ad4ece95653f3dbd97d78ac4d6301bc9042`
  (`Publish SenseCV project source`), and local `master` tracks
  `origin/master`.
- Verification: `python -m py_compile` passed for `src/sensecv/app.py` and
  the scripts using an ignored temporary cache; remote `refs/heads/master`
  resolved to `8d3c6ad4ece95653f3dbd97d78ac4d6301bc9042`.
- Pages touched: [[github-repository]], [[log]].

## [2026-06-06] merge | unified DroNet source/wiki into PilotGuru
- Vendored the durable DroNet source layer from `C:\Users\Isaque\Desktop\dronet`
  into `dronet/`: PyTorch model port, helper scripts, paper sources, and
  upstream `repo/` including `model/model_weights.h5`.
- Updated `app.py`, `run_samsung_dronet_samples.py`, and
  `run_sensecv_02062026_dronet_3fps.py` so `DRONET_DIR` defaults to the local
  `dronet/` folder while still allowing environment overrides.
- Updated `dronet/run_dronet.py` to use local `exports/` and `dronet_results/`
  defaults instead of absolute Desktop paths.
- Merged the standalone DroNet wiki into this canonical wiki by adding
  [[dronet-overview]], [[source-inventory]], [[dronet-paper]],
  [[model-architecture]], [[preprocessing]], [[inference-pipeline]],
  [[experiment-results]], [[pilotguru-10fps-folder-check]], [[ros-control]],
  and [[open-questions]] to [[index]].
- Pages touched: [[overview]], [[index]], [[log]], [[dronet-overview]],
  [[source-inventory]], [[open-questions]].

## [2026-06-03] deploy | loaded 02-06-2026 IFCE example dataset on Fly
- Uploaded the local `SenseCV-02-06-2026-IFCE-Gimbal` dataset to the Fly
  volume through `/api/upload-zip` as the example dataset.
- Fixed duplicate clip discovery when `SENSECV_EXTRA_CLIP_ROOTS` and
  `SENSECV_UPLOADS_DIR` point at the same `/data/clips` root. Uploaded
  recursive datasets now display as `SenseCV-.../01` instead of
  `clips/SenseCV-.../01`.
- Deployed new image:
  `sensecv-api:deployment-01KT788C7KJBMCBXJMYGPPXCJE`.
- Verified `https://sensecv-api.fly.dev/health` returned HTTP 200 with
  `{status:"ok", clips:32}`.
- Verified `/api/clips` starts with
  `SenseCV-02-06-2026-IFCE-Gimbal/01`; `/api/data/0` returns 202 frame
  samples and 57 active external input frames.
- Pages touched: [[deployment-operations]], [[log]].

## [2026-06-03] fix | Fly no-dataset startup and warm machine
- Fixed the viewer boot path for source-only Fly deployments with zero clips.
  `templates/index.html` now renders a no-dataset state and keeps the zip
  import controls available instead of calling `loadClip(0)`.
- Updated `fly.toml` to `min_machines_running = 1` so `sensecv-api` keeps one
  machine warm during interactive testing.
- Deployed new image:
  `sensecv-api:deployment-01KT77HJCXDTV5X7V7BKWNN456`.
- Verified machine `0807567b062618` is version `3`, state `started`;
  `/health` returns HTTP 200 with `{status:"ok", clips:0}`; the served root
  HTML includes `showNoClipsState`.
- Pages touched: [[deployment-operations]], [[log]].

## [2026-06-03] deploy | updated Fly app to latest source image
- Deployed `sensecv-api` to Fly.io with `flyctl deploy -a sensecv-api`.
- New image:
  `sensecv-api:deployment-01KT76K9AY62E3FT637KJ9N7SV`.
- Machine `0807567b062618` is version `2` and reached `started` state.
- Verified `https://sensecv-api.fly.dev/health` returned HTTP 200 with
  `{status:"ok", clips:0}`. This is expected for the source-only image before
  importing datasets onto `/data/clips`.
- Verified `/api/clips` returned an empty clip list and `/api/upload-zip`
  returned the controlled missing-zip error.
- Pages touched: [[deployment-operations]], [[log]].

## [2026-06-03] docs | GitHub remote and current SenseCV project state
- Captured the GitHub project state in [[github-repository]]: local branch
  `master`, remote `origin https://github.com/IsaqueEfraimIFCE/sensecv.git`,
  GitHub account `IsaqueEfraimIFCE`, and the latest committed hashes known at
  the time of documentation.
- Documented the source-only policy for the GitHub repository: datasets, MP4s,
  generated DroNet outputs, upload folders, exports, and local runtime files
  stay out of Git through `.gitignore` / `.dockerignore`.
- Normalized Fly deployment docs to the real lowercase app name
  `sensecv-api` and public URL `https://sensecv-api.fly.dev/`.
- Pages touched: [[overview]], [[index]], [[github-repository]],
  [[deployment-operations]], [[api-routes]], [[log]].

## [2026-06-03] feature | live DroNet classification in Flask viewer
- Added `/api/dronet/<idx>?time=<seconds>&exact=<0|1>` to `app.py`.
  `exact=1` classifies the nearest paused frame; `exact=0` buckets time to
  3 FPS for playback updates.
- Backend lazily loads the local Desktop DroNet PyTorch port and H5 weights
  on first inference request, then caches runtime state and up to 300
  frame-level predictions keyed by clip, video mtime, and frame index.
- Added a compact DroNet panel to `templates/index.html` under the live sensor
  values. It shows direction/steering, yaw degrees, collision probability, and
  the source frame/time. Collision probability >= 0.5 gets red emphasis.
- Frontend behavior: initial clip load and paused/scrubbed frames request exact
  inference; playback requests are deduped to one request per 3 FPS bucket.
- Verified `python -m py_compile app.py`, Flask test client for
  `/api/dronet/0?time=0&exact=1`, and live server on
  `http://127.0.0.1:5001` because port 5000 had a stale/nonmatching process.
- Pages touched: [[overview]], [[index]], [[app-backend]], [[api-routes]],
  [[viewer-frontend]], [[dronet-live-classification]],
  [[dronet-sensecv-02062026-3fps]], [[log]].

## [2026-06-03] ingest | DroNet 3 FPS classifications for SenseCV 02-06-2026 gimbal
- Added `run_sensecv_02062026_dronet_3fps.py`, which imports the local DroNet
  model from `PilotGuru\dronet`, samples every
  `SenseCV-02-06-2026-IFCE-Gimbal` clip at 3 FPS, and writes annotated PNG
  classifications.
- Latest run processed 32 clip folders and wrote 666 frame classifications to
  `dronet_sensecv_02062026_3fps/`.
- Each output subfolder contains `classifications.csv`, annotated
  `classification_*.png` images, and a `contact_sheet.png` with all generated
  images for that clip.
- Aggregate outputs are `all_classifications.csv` and `summary.json`.
- Highest mean collision clips were `21`, `25`, `26`, `22`, `32`, `23`, `24`,
  and `31`; these remain DroNet inspection scores, not validated indoor labels.
- Pages touched: [[overview]], [[index]], [[dronet-sensecv-02062026-3fps]],
  [[log]].

## [2026-06-03] feature | GitHub-ready SenseCV app, zip dataset import, Fly rename
- Added GitHub-facing project files: `README.md`, `.gitignore`,
  `requirements.txt`, `Dockerfile`, `fly.toml`, and `.dockerignore`.
- Renamed the application identity to SenseCV in app configuration, Fly app
  naming, and wiki prose.
- Added `/api/upload-zip` and the "Importar dataset SenseCV" UI module for
  importing zip archives whose nested folders match the SenseCV clip schema.
- Zip import safely rejects path traversal, validates that at least one valid
  clip folder exists, extracts into `SENSECV_UPLOADS_DIR`, refreshes discovery,
  and returns updated clip/group lists.
- Deployed the renamed Fly app `sensecv-api`, verified `/health`, `/api/clips`,
  `/api/data/0`, and the upload endpoint's controlled empty-upload error.
- Pages touched: [[overview]], [[index]], [[app-backend]], [[api-routes]],
  [[viewer-frontend]], [[deployment-operations]], [[log]].

## [2026-05-31] ingest | Samsung/Poco 30-05-2026 gimbal roots and Samsung DroNet samples
- `app.py`: added the inner Downloads roots for
  `SenseCV-30-05-2026-IFCE-Gimbal-Samsung` and
  `SenseCV-30-05-2026-IFCE-Gimbal-Poco` to `EXTRA_CLIP_ROOTS`.
- Verified app discovery at 305 clips total: 48 Samsung clips and 54 Poco clips
  from the new roots. Both join `WALKING_ONLY` through the existing extra-root
  path.
- Read `Coleta  IFCE - 30 de maio de 2026 - ANDRÉ (Samsung) - Página1.csv`.
  It has 53 physical rows but 48 non-empty unique IDs (`1`-`48`), matching the
  Samsung clip folders.
- Added `run_samsung_dronet_samples.py`, which imports the local DroNet model
  from `PilotGuru\dronet`, samples 20 deterministic random frames
  per Samsung ID folder, and writes annotated PNGs plus CSV/JSON summaries.
- Latest run wrote 960 predictions to `dronet_samsung_random_samples/`; all 48
  Samsung clips decoded successfully.
- Pages touched: [[overview]], [[index]], [[app-backend]], [[SenseCV-dataset]],
  [[dronet-samsung-samples]], [[log]].

## [2026-05-30] feature | export-aware clip selection, clean export MP4s, live state
- `app.py`: clip discovery now includes `exports/` as selectable clips. Raw
  clips use `video.mp4`; exports use `<folder>.mp4`, so `CLIP_VIDEO_PATHS`
  stores the exact MP4 path. `CLIP_GROUPS` drives source grouping in the UI.
- `templates/index.html`: added a folder/group selector that jumps directly to
  the first clip in a group, while the clip selector uses optgroups. Groups now
  include `Supermercado Telefrango`, each SenseCV root, and `exports`.
- Auto-suggest now tries `lateral` first, then `walking`, then `vertical`.
- `history.json` is pruned against actual export folders on load. Next export
  number comes from the first missing leading number in `exports/`; no numbered
  exports means `01`.
- Added `/api/export-state` polling to refresh clips, groups, history, export
  folders, and next number without restarting.
- `run_server.bat` / `run_server.ps1`: stop existing port-5000 listeners before
  launching this checkout's `.venv` Python and `app.py`, preventing stale Flask
  processes from serving old templates.
- Export video cutting now re-encodes with H.264 (`libx264`) instead of
  `-c copy`, preserving rotation metadata and fixing short/keyframe-broken
  exported clips. Repaired 24 history-backed exports; two folders without
  history records (`06_obstaculo_esquerda_desvio_direita`, `erro4`) were left
  unchanged.
- Pages touched: [[overview]], [[index]], [[app-backend]], [[viewer-frontend]],
  [[api-routes]], [[clip-data-model]], [[exports-output]], [[history-json]],
  [[crop-suggestion]], [[export-pipeline]], [[log]].

## [2026-05-28] ingest | new SenseCV source root: SenseCV-27-05-2026-IFCE-Gimbal (24 clips)
- Added `C:\Users\Isaque\Desktop\SenseCV-27-05-2026-IFCE-Gimbal` as a 4th entry
  in `EXTRA_CLIP_ROOTS` (`app.py:21`). Gimbal-stabilized variant of the
  27-05-2026 footage.
- All 24 subfolders (`01`–`24`) validated as clip dirs per `_is_clip_dir`
  (each has `video.mp4` + `frames.json` + the usual sensor/extra JSONs).
- Surfaces in the UI with the `SenseCV-27-05-2026-IFCE-Gimbal/NN` display
  prefix and joins `WALKING_ONLY` (horizontal footage — vertical-orientation
  detection skipped, per [[orientation-detection]]).
- Total clip count after restart: **281** (was 257; +24). Verified via
  `/api/clips`.
- pages touched: [[log]]

## [2026-05-28] ingest | new SenseCV source root: SenseCV-27-05-2026-IFCE (33 clips)
- Extracted `SenseCV-27-05-2026-IFCE.zip` to
  `C:\Users\Isaque\Downloads\SenseCV-27-05-2026-IFCE\` (single level; zip
  already wrapped the folder, unlike the 2023 root's double-nested path).
- All 33 subfolders (`01`–`33`-ish) validated as clip dirs per `_is_clip_dir`
  in `app.py` (each has `video.mp4` + `frames.json` + `accelerations.json` +
  `rotations.json`).
- Wired into `EXTRA_CLIP_ROOTS` in `app.py:17-20`; clips will surface in the
  UI with the `SenseCV-27-05-2026-IFCE/NN` display prefix and join
  `WALKING_ONLY` (horizontal footage — vertical-orientation detection
  skipped, per [[orientation-detection]]).
- Total clip count: 157 supermercado + 67 (23-05-2023) + 36 (26-05-2026) +
  33 (27-05-2026) = **293** (was 260 before this ingest).
- Each clip also ships `gps_status.json`, `locations.json`,
  `magnetometer.json`, and a stub `external_sensors.json` — none of these are
  consumed by the current pipeline (only video / frames / accel / rotations
  are read), so [[clip-data-model]] still reflects the system's view; the
  extras are noise to it.
- Pages touched: [[log]] (this entry); [[app-backend]] unchanged in behavior
  but its `EXTRA_CLIP_ROOTS` list now has a third entry.

## [2026-05-28] redesign | lateral detector now scores mean |lateral velocity|
- Per user request, swapped the lateral detector from yaw rate to mean
  |lateral velocity| over a sliding 1 s window. Lateral velocity = integrated
  horizontal accel, decomposed against the local rolling-mean walking
  direction; the perpendicular component is the lateral one.
- Benchmark against the same 5 GT clips: lateral-velocity scored 0.61 s mean
  abs offset, slightly better than yaw rate's 0.68 s and comparable to a
  z(yaw)+z(lat-vel) combination (0.53 s — kept out for simplicity).
- Pages touched: [[crop-suggestion]], [[log]].

## [2026-05-28] redesign | lateral detector switched from accel-mag to yaw rate
- Calibrated `suggest_lateral_deviation()` against the operator's 5 deviation
  exports in [[history-json]] (entries 6-10, SenseCV clips 28/29/30/31/46).
- Debug script `_debug_lateral.py` showed lateral-accel magnitude is a poor
  feature: for 3 of 5 GT clips the global lateral-mag max lay outside the
  labeled window, and a sliding 1 s mean over it also picked the wrong window.
  Yaw rate (gyro component along the gravity axis = body pivoting) matched
  all 5 GT windows within ~0.5 s.
- Rewrote the detector around a sliding 1 s window scored by mean yaw rate,
  always returning a window (per the user's "must happen in every lateral
  deviation clip" requirement). Best match: clip 197 within 0.03 s; mean
  absolute start offset 0.30 s, end offset 0.38 s.
- `SenseCV dataset (lateral)/` is now stale again — re-run the lateral
  batches via the UI to refresh.
- Pages touched: [[crop-suggestion]], [[log]].

## [2026-05-27] tune | lateral detector: score by peak, not length
- `suggest_lateral_deviation()` now scores candidate runs by **peak magnitude**
  instead of duration, raises the bar to a two-tier threshold (sustain 1.5 +
  peak 3.5 m/s²), and trims the kept run to a ≤ 1.2 s window centered on the
  peak. Reasoning: an obstacle-avoidance sidestep is a sharp, brief impulse;
  the prior "longest above 0.8 m/s²" rule kept slowly drifting low-amplitude
  windows instead.
- Reduced pre-smoothing in `_lateral_acceleration_series()` from ~0.3 s to
  ~0.1 s so the impulse peak isn't flattened out before scoring.
- Spot-checks: supermercado clip 0 went from 1.6–12.4 s (11 s of mild motion)
  to a 1.1 s window centered on its true peak at 25.9 s. SenseCV walking-only
  clips that previously matched the loose threshold now correctly report no
  detection.
- The previous `SenseCV dataset (lateral)/` run (102 ok) is now stale; re-run
  the SenseCV / Supermercado lateral batches via the UI to refresh.
- Pages touched: [[crop-suggestion]], [[log]].

## [2026-05-27] feature | UI batch export + lateral-deviation cut mode
- `app.py`: added `suggest_lateral_deviation()` — single longest sustained
  horizontal-accel burst (smoothed 0.3 s, threshold 0.8 m/s², minimum 0.2 s),
  with gravity removed via 1 s rolling-mean so it works in any phone
  orientation. Wired through `/api/suggest?mode=lateral`.
- `app.py`: moved `export_set()` + preset helpers into the module so the
  Flask process and the CLI share one code path (avoids the circular-import
  trap that `from export_SenseCV import export_set` would have triggered).
  Added `POST /api/batch-export` (synchronous; returns ok/skipped/failed).
- `templates/index.html`: third "⬦ Desvio lateral" suggest button; new
  "Exportação em lote" panel with 4 preset×mode buttons; generalized
  `runSuggest` via `SUGGEST_BTN_IDS` / `SUGGEST_LABELS` maps.
- `export_SenseCV.py`: collapsed into a thin CLI around `app.export_set`,
  accepts `[preset] [mode]` args. Lateral output goes to
  `<base> dataset (lateral)/` to avoid clashing with walking cuts.
- Verified end-to-end via CLI `SenseCV lateral`: 102 ok, 1 below-threshold
  skip, 0 failed.
- Pages touched: [[crop-suggestion]], [[api-routes]], [[viewer-frontend]],
  [[app-backend]], [[SenseCV-dataset]].

## [2026-05-27] feature | Supermercado dataset batch export (sibling to SenseCV)
- Refactored `export_SenseCV.py` around `export_set(indices, out_dir)` driven by
  two `PRESETS` (`SenseCV` default, `supermarket` opt-in via argv). The
  supermarket preset selects every clip *not* in `WALKING_ONLY` and writes to
  `Supermercado dataset/`, with its own `sources.csv`.
- Calls `suggest_crop(idx, 'walking')` explicitly so supermercado clips go
  through the learned-window / Random Forest path ([[crop-suggestion]]) rather
  than the broader vertical mode.
- Last run: 71 ok, 86 no-walking, 0 failed across the 157 supermercado clips —
  same coverage as the UI's auto-suggest.
- Also fixed a Windows cp1252 stdout crash from a `→` literal in the progress
  print.
- Pages touched: [[SenseCV-dataset]] (broadened to cover both datasets),
  [[index]], [[log]].

## [2026-05-27] feature | SenseCV dataset batch export + provenance CSV
- Added `export_SenseCV.py`: iterates every clip in `WALKING_ONLY`, runs the
  horizontal-branch `suggest_crop()` ([[crop-suggestion]]), ffmpeg-cuts the
  walking segment with `-c copy`, and reuses `save_sensor_data()` for the
  rebased per-frame/sensor JSON ([[export-pipeline]]).
- Outputs land in `SenseCV dataset/<source-prefixed-name>/` (e.g.
  `SenseCV-23-05-2023-IFCE_06/`), with a `sources.csv` ledger mapping each
  output back to its source root, subfolder, absolute path, and crop window.
- Deliberately bypasses `history.json` so SenseCV cuts do not enter the walking
  classifier's training set — keeps the horizontal regime out of the
  supermarket suggestion loop. See [[SenseCV-dataset]].
- Pages touched: [[SenseCV-dataset]] (new), [[index]], [[log]].

## [2026-05-27] feature | multi-root clip input + horizontal walking-only clips
- `app.py`: clip discovery now scans `CLIPS_DIR` plus `EXTRA_CLIP_ROOTS`
  (two `SenseCV-*` folders). Added `CLIP_PATHS` (name→abs path) and
  `clip_path()` resolves through it; extra-root clips get a `label/name`
  display name to stay unique (their subfolders are just `01`,`02`,…). Primary
  root clips keep bare names, so existing history/exports stay valid.
- `app.py`: added `WALKING_ONLY` set. `suggest_crop()` branches early for those
  clips — ignores the vertical mask and segments the first sustained walking run
  (`_first_sustained(walking,…)`), honoring learned windows; returns
  `has_vertical:true` so the warning never fires. Only the supermarket (primary)
  root still uses vertical-orientation detection.
- Clip count 157 → 260 (+67 +36). Verified: SenseCV clip 157 → walking 0.0–9.07,
  supermarket clip 0 → learned 18.999–25.62.
- Pages touched: [[crop-suggestion]], [[orientation-detection]], [[app-backend]],
  [[clip-data-model]].

## [2026-05-26] feature | learned walking crop suggestions from exports
- `app.py`: added `learned_walking_window()` so walking suggestions use the
  latest exported crop for a source clip before falling back to sensor
  heuristics; added `_best_sustained()` for new clips without export history.
- `templates/index.html`: auto-suggest now requests `walking` mode instead of
  the broader vertical-only mode.
- Verified the five current `history.json` exports match walking suggestions
  exactly.
- Pages touched: [[crop-suggestion]], [[app-backend]], [[viewer-frontend]].

## [2026-05-26] feature | benchmark-based walking fallback
- `app.py`: replaced the generic walking fallback with a benchmark scorer built
  from exported crop windows and nearby non-crop context; unexported clips now
  use `_benchmark_profile()` + `_benchmark_sustained()`.
- Hardened suggestion handling for short/malformed clip sensor files so the API
  returns a controlled no-suggestion response instead of raising.
- Verified all 157 clips: 95 suggestions, 62 controlled no-suggestion responses,
  0 crashes.
- Pages touched: [[crop-suggestion]], [[app-backend]].

## [2026-05-26] feature | template-matched single walking window
- `app.py`: changed unexported walking suggestions from frame scoring to sliding
  sequence matching against the exported ground-truth windows. The fallback now
  returns one candidate window per clip: the segment whose smoothed sensor
  sequence is closest to the exported walking templates.
- Added a `history.json` mtime cache for the walking templates so new exports
  refresh the benchmark automatically.
- Verified all 157 clips: 111 suggestions, 46 controlled no-suggestion
  responses, 0 crashes.
- Pages touched: [[crop-suggestion]], [[app-backend]].

## [2026-05-26] feature | IMU/Gyroscope walking detector
- `app.py`: replaced the active walking fallback with `_imu_walking_window()`,
  which predicts walking from accelerometer high-pass energy, gyroscope energy,
  jerk, and portrait gating.
- Calibrated against the five exported ground-truth windows; learned exports
  still return exact saved windows, while unexported clips use the IMU detector.
- Full scan: 157 clips, 88 walking suggestions, 69 controlled no-suggestion
  responses, 0 invalid-duration suggestions.
- Pages touched: [[crop-suggestion]], [[app-backend]].

## [2026-05-26] feature | supervised walking classifier
- `app.py`: added a cached Random Forest frame classifier trained from
  `history.json` exports, using IMU/gyroscope energy plus rolling gait-cadence
  autocorrelation features.
- `suggest_crop(..., mode='walking')`: unexported clips now use the classifier
  fallback, then probability smoothing and one-window segment selection.
- Validation: exact exported clips still return saved windows; classifier-only
  benchmark error on the 5 labeled clips is below 0.25 s per clip; full scan
  found 71 valid windows, 86 controlled no-suggestion responses, 0 bad
  durations/crashes.
- Pages touched: [[crop-suggestion]], [[app-backend]].

## [2026-05-26] feature | auto-suggest, vertical warning, synced playback, walking mode
- Code change ingested from a session that modified `app.py` + `templates/index.html`.
- `app.py`: refactored `suggest_crop` into `_orientation_walking_masks()` +
  `_first_sustained()`; added `mode` param (`vertical`/`walking`) and
  `has_vertical` to the response; `/api/suggest` now reads `?mode=`.
- `templates/index.html`: auto-suggest on clip load, "no vertical moment"
  warning banner, play/pause button + `requestAnimationFrame` sync loop, second
  "Vertical + andando" suggest button via `runSuggest(mode)` with stale guard.
- Verified on real clips: clip0 vertical 14.23→25.62, walking 22.39→25.62;
  clip2 has_vertical=false (warning case).

## [2026-05-26] init | wiki created
- Bootstrapped the LLM Wiki for SenseCV from the codebase (`app.py`,
  `templates/index.html`) as raw source.
- Wrote schema (`CLAUDE.md`), [[index]], [[overview]], and 13 entity/concept
  pages covering the backend, frontend, data model, API, and the sensor /
  crop / export algorithms.
- Pages touched: all (initial creation). Page count: 16.
- Open questions filed: whether [[walking-detection]] should use fused `speed`
  from [[velocity-estimation]]; frame-exact vs keyframe-copy cut tradeoff in
  [[export-pipeline]].

