---
type: entity
tags: [deployment, fly, docker, api]
code_refs: [Dockerfile, fly.toml, requirements.txt, app.py]
updated: 2026-06-03
---

# Deployment and operations

SenseCV can run as a Flask API/viewer on Fly.io using the root-level
`Dockerfile` and `fly.toml`. This follows the same production shape documented
for FilterTrack: Docker image, Fly `http_service` on internal port `8080`,
HTTPS forced, single app process in region `gru`, and a Fly volume mounted at
`/data`.

## Fly configuration

```text
App: sensecv-api
Region: gru
Internal port: 8080
Volume source: sensecv_data
Volume destination: /data
```

Environment in `fly.toml`:

| variable | purpose |
|---|---|
| `PORT=8080` | Gunicorn bind port; must match `http_service.internal_port` |
| `SENSECV_EXTRA_CLIP_ROOTS=/data/clips` | Mounted dataset root scanned by `find_clips()` |
| `SENSECV_EXPORTS_DIR=/data/exports` | Persistent interactive exports |
| `SENSECV_HISTORY_FILE=/data/history.json` | Persistent export history |
| `SENSECV_UPLOADS_DIR=/data/clips` | Destination for uploaded SenseCV zip datasets |

The service keeps one machine warm for interactive use:

```text
auto_stop_machines = "stop"
auto_start_machines = true
min_machines_running = 1
```

## Deploy

```powershell
flyctl apps create sensecv-api --org personal
flyctl volumes create sensecv_data --region gru --size 20 -a sensecv-api
flyctl deploy -a sensecv-api
```

Useful checks:

```powershell
flyctl status -a sensecv-api
Invoke-WebRequest -Uri "https://sensecv-api.fly.dev/health" -UseBasicParsing
Invoke-WebRequest -Uri "https://sensecv-api.fly.dev/api/clips" -UseBasicParsing
```

`/health` returns `{status:"ok", clips:<count>}` after refreshing clip
discovery.

## Dataset storage

The local checkout is about 14 GB, mostly MP4s, so the GitHub-ready Docker image
intentionally does **not** copy dataset folders. Import datasets through the
viewer zip upload or place source clip folders under:

```text
/data/clips/<dataset>/<clip>/
```

Environment-provided roots and uploaded zip datasets are scanned recursively for
valid clip folders, so nested SenseCV wrappers are accepted.

Each clip folder still needs the normal [[clip-data-model]] files:
`video.mp4` or `<folder>.mp4`, `frames.json`, `accelerations.json`, and
`rotations.json`; optional `external_sensors.json` is exposed by `/api/data`.

## Local container behavior

The container runs:

```text
gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 4 --timeout 180 app:app
```

The Flask dev entrypoint remains available for local use and now honors
`PORT`, defaulting to `5000`.

## Latest observed state

Verified on 2026-06-03:

```text
Public URL: https://sensecv-api.fly.dev/
Health: /health HTTP 200, clips=0
Image: sensecv-api:deployment-01KT77HJCXDTV5X7V7BKWNN456
Machine: 0807567b062618
Machine version: 3
Machine state: started
```

`/api/clips` returns `{clips:[], groups:[], total:0}` for the current
source-only image until a dataset is imported onto the Fly volume.
The root viewer initializes in a no-dataset state instead of calling
`loadClip(0)`, so the zip import panel remains usable with zero clips.
`/api/upload-zip` is live and returns a controlled 400 when no zip file is
provided.

The current GitHub-ready source image excludes bundled datasets. Import
datasets through `/api/upload-zip` or place valid clip folders under
`/data/clips`.


