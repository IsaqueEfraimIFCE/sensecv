---
type: entity
tags: [backend, api, http]
code_refs: [app.py]
updated: 2026-06-15
---

# API routes

Endpoints verified in the current recovered checkout. Clips are addressed by
integer index into the current `CLIPS` list.

| Method | Path | Returns | Purpose |
|---|---|---|---|
| GET | `/` | HTML | Refreshes clip discovery and renders [[viewer-frontend]] with bootstrap data |
| GET | `/health` | JSON | `{status:"ok", clips:<count>}` after `refresh_clips()`; intended for Fly health checks |
| GET | `/video/<int:idx>` | video/mp4 | Streams the clip's actual MP4 path; honors HTTP `Range` for seeking |
| GET | `/frame/<int:idx>/<int:source_index>.jpg` | image/jpeg | Decodes one source frame as a thumbnail for SSIM frame review |
| GET | `/export-file/<folder>/<path:relpath>` | file | Serves generated export files from an export folder |
| GET | `/api/data/<int:idx>` | JSON | `{fps, duration, times[], accel[], rotation[], velocity[], external_input[], quality, name, index, total}`; cached |
| GET | `/api/dronet/<int:idx>?time=&exact=` | JSON | Live DroNet steering/yaw/collision for a requested video time. See [[dronet-live-classification]] |
| GET | `/api/sensemodel/<int:idx>?time=&exact=` | JSON | Live two-head SenseCV `.keras` obstacle/deviation prediction for a requested frame |
| GET | `/api/clips` | JSON | `{clips[], groups[], total}` after `refresh_clips()` |
| POST | `/api/upload-zip` | JSON | Imports a SenseCV-style `.zip`, refreshes clips, returns `{status,dataset,clips_added,clips,groups,total}` |
| GET | `/api/history` | JSON | Pruned contents of [[history-json]] |
| GET | `/api/export-state` | JSON | Refreshes clips and returns clips/groups/history/export folders/next number |
| GET | `/api/next-number` | JSON | `{number}` from actual export folders |
| GET | `/api/suggest/<int:idx>?mode=` | JSON | Crop proposal; `mode` in `vertical` / `walking` / `lateral`. See [[crop-suggestion]] |
| GET | `/api/ssim/<int:idx>?start=&end=` | JSON | Full SSIM frame selection for a crop, including selected frame metadata and thumbnail URLs |
| GET | `/api/imu-events/<int:idx>?delta=` | JSON | IMU capture-quality verdict plus desvio/reducao/parada events with T1/T2, label windows, and confidence. See [[imu-event-labeling]] |
| POST | `/api/crop` | JSON | Performs the export. See [[export-pipeline]] |
| POST | `/api/batch-export` | JSON | Runs `export_set()` over `{preset:'all'|'sensecv'|'supermarket', mode:'walking'|'lateral'}` and returns CSV/review links |
| POST | `/api/manifest-export` | JSON | Plans the uploaded-datasets export queue: every clip whose dataset ships an `.xlsx` manifest. Resets `data/derived/manifest_exports/` by default |
| POST | `/api/manifest-export/clip` | JSON | Exports exactly one queued clip (`{clip_idx, mode}`), so failures identify the specific source clip |
| POST | `/api/manifest-export/review` | JSON | Builds one review MP4 from all successful manifest cuts, in queue order |

## `/api/sensemodel`
Live inference for the local two-head SenseCV `.keras` model. The model path
defaults to `best_model.keras` and is overridable with `SENSECV_MODEL_PATH`.
Heads are matched by output width (2 → obstacle `NONE/OBSTACLE`, 3 → deviation
`LEFT/RIGHT/NONE`); the frame is resized to the model's own input shape. The
exported model is **MobileNetV2 with an embedded `Rescaling` layer**
(`scale=1/127.5, offset=-1`), so the route feeds **raw 0–255 RGB pixels** — do
NOT divide by 255 first, or the input range collapses and every frame yields a
near-constant prediction. Color is BGR→RGB; input is 224×224×3. The
response carries `obstacle_label`, `obstacle_prob`, `deviation_label`,
`deviation_prob`, and the raw `obstacle_probs` / `deviation_probs` vectors. Like
DroNet, it lazy-loads TensorFlow and degrades gracefully — a missing model file
or absent TensorFlow returns `{available:false, error}` with 503 rather than
crashing. **Note:** this restoration has no `.keras` model file, so the route
currently always returns the controlled error until a model is provided.

## `/api/manifest-export`
Re-implemented from the documented behavior after the original Desktop code was
lost (see [[log]] 2026-06-15). The planning call walks `CLIPS`, finds each
clip's top-level uploaded dataset under `UPLOADS_DIR`, looks for any `.xlsx`
there, parses it with `openpyxl` (first sheet containing an `ID` column), and
matches the clip's zero-padded folder name to a manifest `ID`. Matched clips
become queue items `{clip_idx, source_display, group, manifest_id, label}` where
`label` joins `ID | DESCRICAO | POSICAO CELULAR | LOCAL OBSTACULO | ALTURA
OBSTACULO` (see [[sensecv-02062026-ifce-clip-manifest]]).

```jsonc
{ "status": "ok", "mode": "lateral", "total": 3,
  "items": [ { "clip_idx": 0,
               "source_display": "SenseCV-02-06-2026-IFCE/.../01",
               "group": "SenseCV-02-06-2026-IFCE", "manifest_id": "01",
               "label": "01 | lixeira azul | celular deitado | direita" } ] }
```

`/api/manifest-export/clip` cuts one queued clip with the same auto-suggested
window as the viewer (`lateral` → `suggest_lateral_deviation`, else
`walking`), saves time-rebased sensors and the SSIM selection, and appends a row
to a per-group `sources.csv`. Outputs land under
`data/derived/manifest_exports/<group>/<mode>/<clip>/`, never under `exports/`,
so they don't touch [[history-json]]. A `found:false` suggestion returns
`status:"skipped"`; a real cut failure returns 500 with `source_display`.
`/api/manifest-export/review` concatenates every successful cut (re-planned in
queue order) into `data/derived/manifest_exports/all_uploaded_<mode>_review.mp4`.
Requires `openpyxl` (added to `requirements.txt`); without a manifest the queue
is simply empty.

## `/api/export-state`
Used by [[viewer-frontend]] polling every 5 seconds.

```jsonc
{
  "clips": ["2026_01_16-12_44_51", "exports/01_obstaculo_..."],
  "groups": ["Supermercado Telefrango", "exports"],
  "total": 307,
  "history": [],
  "export_folders": ["01_obstaculo_..."],
  "next_number": 6
}
```

This endpoint calls `refresh_clips()` and `load_history()`. `load_history()`
prunes records whose `folder` no longer exists under `exports/`.

## `/api/data` external input
If a clip folder contains `external_sensors.json`, the backend returns
`external_input[]` aligned one-to-one with `times[]` / video frames. Each value
is `1` while the latest external sensor sample has `button: 1`, otherwise `0`.
Clips without the file return an empty array.

## `/api/data` capture quality

`quality` reports the worst observed sampling interval converted to a rate:

```jsonc
{
  "quality": {
    "video_min_fps": 14.73,
    "imu_min_hz": 420.88,
    "gyro_min_hz": 420.88
  }
}
```

This is a minimum-rate quality indicator, not the nominal median FPS shown in
the older header badge. A video min near 14.7 FPS usually means a brief doubled
frame interval in an otherwise ~29.5 FPS clip.

## `/api/dronet`
Live DroNet inference for the viewer's current frame.

```text
GET /api/dronet/<idx>?time=1.23&exact=1
GET /api/dronet/<idx>?time=1.23&exact=0
```

`exact=1` maps `time` directly to the nearest source video frame. `exact=0`
buckets time to 3 FPS (`floor(time * 3) / 3`) before choosing the frame. The
frontend uses exact mode for paused/scrubbed frames and bucketed mode while
playing.

```jsonc
{ "available": true, "frame": 30, "time_s": 0.9993,
  "requested_time_s": 1.23, "sample_fps": 3.0, "exact": false,
  "source_fps": 30.0213, "steering": 0.0577, "yaw_deg": 5.1956,
  "direction": "STRAIGHT", "collision_prob": 0.9995,
  "collision_label": "COLLISION" }
```

Failures return `{available:false,error}` with 503, except out-of-range clip
indexes return 404.

## `/api/suggest` response
```jsonc
// found
{ "found": true, "mode": "lateral", "start": 4.24, "end": 5.23, "has_vertical": true,
  "ssim": { "frames_before": 31, "frames_after": 12, "ssim_threshold": 0.985 } }
// not found
{ "found": false, "mode": "walking", "has_vertical": true,
  "message": "Posicao vertical encontrada, mas sem momento de caminhada sustentado" }
```

`has_vertical:false` drives the UI warning. `mode='lateral'` returns
`has_vertical:true` because orientation is irrelevant for the lateral detector.
For found cuts, `ssim.frames_before` is the frame count inside the proposed
window and `ssim.frames_after` is the visually distinct image count selected by
the SSIM de-duplication pass.

## `/api/ssim`
Returns the complete frame selection for the current crop window. The viewer
uses it to render a thumbnail strip after suggestions and when the user clicks
the refresh button for a manually marked interval.

Both `/api/ssim` and `/api/suggest` accept `?metric=ssim|dinov2|lpips|vif`
(default `ssim`); `/api/crop` accepts `ssim_metric` in the body. When
`threshold` is omitted, the metric's default applies (SSIM
`SENSECV_SSIM_THRESHOLD`, DINOv2 `0.98`, LPIPS `0.95`, VIF `0.70` — see
`DEFAULT_METRIC_THRESHOLDS` in `app.py`). The non-SSIM backends are
lazy-loaded (`torch.hub` DINOv2 ViT-S/14, `lpips` AlexNet, `sewar` VIF) and
the score is still reported as `ssim_prev` / `ssim_threshold` for
compatibility, with the chosen `metric` echoed in the payload.

```jsonc
{
  "frames_before": 31,
  "frames_after": 12,
  "metric": "ssim",
  "ssim_threshold": 0.985,
  "ssim_max_gap_sec": 0.5,
  "selected_frames": [
    { "source_index": 120, "frame_id": 120, "time_s": 4.0,
      "ssim_prev": null, "image_url": "/frame/0/120.jpg" }
  ]
}
```

## `/api/batch-export`
Synchronous: the response returns after every clip has been processed.

```jsonc
{ "preset": "sensecv", "mode": "lateral" }
```

```jsonc
{ "status": "ok", "preset": "sensecv", "mode": "lateral",
  "ok": 96, "skipped": 7, "failed": 0, "total": 103,
  "out_dir": "...\\SenseCV dataset (lateral)",
  "csv_path": "...\\SenseCV dataset (lateral)\\sources.csv",
  "csv_url": "/derived-file/SenseCV%20dataset%20%28lateral%29/sources.csv",
  "review": {
    "url": "/derived-file/SenseCV%20dataset%20%28lateral%29/review_all_frames.mp4",
    "frames": 2880,
    "videos": 96
  } }
```

Invalid preset/mode returns 400. Presets exclude `exports/` display names even
though those clips are selectable in the viewer. `preset:"all"` processes all
non-export clips. The review video is a single MP4 containing every frame from
the exported cut videos, intended for quick visual inspection.

## `/api/crop` request / response
```jsonc
{ "clip_idx": 0, "start": 14.23, "end": 25.62, "name": "01",
  "occurrence": "obstaculo", "obs_pos": "centro",
  "response": "desvio", "desvio_dir": "direita" }
```

```jsonc
{ "status": "ok", "file": "01_..._direita.mp4",
  "folder": "01_..._direita",
  "export_folders": ["01_..._direita"],
  "next_number": 2 }
```

Errors return `{status:"error", message}` with 400 (empty name), 409
(collision), or 500 (ffmpeg/IO failure). See [[classification-taxonomy]] for how
the fields compose the folder name.
