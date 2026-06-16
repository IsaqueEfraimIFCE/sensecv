---
type: entity
tags: [deployment, fly, docker, api]
code_refs: [Dockerfile, fly.toml, requirements.txt, app.py]
updated: 2026-06-12
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

Verified on 2026-06-12:

```text
Public URL: https://sensecv-api.fly.dev/
Health: /health HTTP 200, clips=32
Image: sensecv-api:deployment-01KTY1JEYZ... (version 6)
Machine: 0807567b062618
Machine state: started
```

Frame-selection metrics on Fly: SSIM and VIF work (`sewar` is in
`requirements.txt`); DINOv2 and LPIPS return a controlled
`metric ... indisponivel: No module named 'torch'` error in the selection
payload because the slim image deliberately excludes torch (~2 GB and the
shared-1GB VM would likely OOM). Use the local server for those metrics.

## Gotchas

- `deploy/fly.toml` `[build].dockerfile` is resolved **relative to the config
  file**, so it must be `"Dockerfile"`, not `"deploy/Dockerfile"`. Deploy from
  the repo root with `flyctl deploy -c deploy/fly.toml -a sensecv-api` (the
  repo root stays the build context for `COPY src ./src`).
- With `SENSECV_DATA_DIR=/data`, `CLIPS_DIR` defaults to `/data/clips`, the
  same path as `SENSECV_UPLOADS_DIR`. `find_clips()` must scan that root
  recursively or nested uploaded datasets are invisible (`clips=0`); fixed on
  2026-06-12 in `app.py › find_clips()`.

`/api/clips` returns 32 example clips from
`SenseCV-02-06-2026-IFCE-Gimbal/01` through `/32`, grouped as
`SenseCV-02-06-2026-IFCE-Gimbal`.
`/api/data/0` returns 202 `times`, 202 `accel`, and 202 `external_input`
samples, with 57 active input frames.
`/api/upload-zip` is live and returns a controlled 400 when no zip file is
provided.

The current GitHub-ready source image still excludes bundled datasets. The
02-06-2026 IFCE example dataset lives on the Fly volume under `/data/clips`;
future datasets can be imported through `/api/upload-zip` or placed as valid
clip folders under `/data/clips`.


