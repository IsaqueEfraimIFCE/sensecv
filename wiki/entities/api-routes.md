---
type: entity
tags: [backend, api, http]
code_refs: [app.py]
updated: 2026-06-02
---

# API routes

All endpoints served by [[app-backend]]. Clips are addressed by integer index
into the current `CLIPS` list.

| Method | Path | Returns | Purpose |
|---|---|---|---|
| GET | `/` | HTML | Refreshes clip discovery and renders [[viewer-frontend]] with bootstrap data |
| GET | `/health` | JSON | `{status:"ok", clips:<count>}` after `refresh_clips()`; intended for Fly health checks |
| GET | `/video/<int:idx>` | video/mp4 | Streams the clip's actual MP4 path; honors HTTP `Range` for seeking |
| GET | `/api/data/<int:idx>` | JSON | `{fps, duration, times[], accel[], rotation[], velocity[], external_input[], name, index, total}`; cached |
| GET | `/api/clips` | JSON | `{clips[], groups[], total}` after `refresh_clips()` |
| POST | `/api/upload-zip` | JSON | Imports a SenseCV-style `.zip`, refreshes clips, returns `{status,dataset,clips_added,clips,groups,total}` |
| GET | `/api/history` | JSON | Pruned contents of [[history-json]] |
| GET | `/api/export-state` | JSON | Refreshes clips and returns clips/groups/history/export folders/next number |
| GET | `/api/next-number` | JSON | `{number}` from actual export folders |
| GET | `/api/suggest/<int:idx>?mode=` | JSON | Crop proposal; `mode` in `vertical` / `walking` / `lateral`. See [[crop-suggestion]] |
| POST | `/api/crop` | JSON | Performs the export. See [[export-pipeline]] |
| POST | `/api/batch-export` | JSON | Runs `export_set()` over `{preset:'SenseCV'|'supermarket', mode:'walking'|'lateral'}` |

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

## `/api/suggest` response
```jsonc
// found
{ "found": true, "mode": "lateral", "start": 4.24, "end": 5.23, "has_vertical": true }
// not found
{ "found": false, "mode": "walking", "has_vertical": true,
  "message": "Posicao vertical encontrada, mas sem momento de caminhada sustentado" }
```

`has_vertical:false` drives the UI warning. `mode='lateral'` returns
`has_vertical:true` because orientation is irrelevant for the lateral detector.

## `/api/batch-export`
Synchronous: the response returns after every clip has been processed.

```jsonc
{ "preset": "SenseCV", "mode": "lateral" }
```

```jsonc
{ "status": "ok", "preset": "SenseCV", "mode": "lateral",
  "ok": 96, "skipped": 7, "failed": 0, "total": 103,
  "out_dir": "...\\SenseCV dataset (lateral)",
  "csv_path": "...\\SenseCV dataset (lateral)\\sources.csv" }
```

Invalid preset/mode returns 400. Presets exclude `exports/` display names even
though those clips are selectable in the viewer.

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


