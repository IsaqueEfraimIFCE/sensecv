# SenseCV

SenseCV is a Flask web app for reviewing recorded walking clips, visualizing
video-synchronized IMU streams, importing SenseCV-style datasets, and exporting
annotated crop windows.

## Features

- Video player with synchronized acceleration, gyroscope, velocity, and external
  input charts.
- SenseCV-style `.zip` dataset upload through `/api/upload-zip`.
- Automatic crop suggestions for walking and lateral-deviation windows.
- Exported clips with rebased `frames.json`, `accelerations.json`,
  `rotations.json`, and optional `external_sensors.json`.
- Fly.io deployment using `Dockerfile` and `fly.toml`.

## Dataset Format

Upload a zip containing one or more clip folders. Nested wrapper folders are
accepted. Each clip folder needs:

```text
video.mp4
frames.json
accelerations.json
rotations.json
```

Optional files such as `external_sensors.json`, `locations.json`,
`gps_status.json`, and `magnetometer.json` may also be present. When
`external_sensors.json` contains `button: 1`, the UI shows an orange `Input`
trace on the acceleration chart.

## Local Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Fly Deploy

```powershell
flyctl apps create sensecv-api --org personal
flyctl volumes create sensecv_data --region gru --size 20 -a sensecv-api
flyctl deploy -a sensecv-api
```

Production URL:

```text
https://sensecv-api.fly.dev/
```

Datasets uploaded in production are stored under `/data/clips` through
`SENSECV_UPLOADS_DIR`.

## Wiki

Project documentation lives in `wiki/`. Start with `wiki/index.md` and
`wiki/overview.md`.
