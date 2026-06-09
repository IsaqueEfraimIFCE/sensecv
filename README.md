# senseCV

senseCV is a Flask application for reviewing recorded walking clips, visualizing synchronized video/IMU streams, importing SenseCV-style datasets, exporting annotated crop windows, and optionally running DroNet inference.

## Project Structure

```text
senseCV/
  src/sensecv/              Flask application package
    app.py                  Web app, APIs, crop/export logic
    templates/index.html    Browser viewer/annotator UI
  scripts/                  Local CLIs and maintenance scripts
  third_party/dronet/       Vendored DroNet runtime, paper sources, upstream repo, weights
  docs/wiki/                Maintained project wiki
  deploy/                   Dockerfile and Fly.io config
  data/                     Local-only datasets, exports, uploads, generated outputs
  logs/                     Local-only server logs
```

`data/` is intentionally ignored by Git. Put clip folders in `data/clips/`, imported/reference datasets in `data/datasets/`, and generated outputs will go to `data/exports/`, `data/derived/`, or `data/dronet_results/`.

## Local Setup

```powershell
cd C:\Users\Isaque\Desktop\senseCV
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For DroNet inference, also install:

```powershell
pip install -r requirements-dronet.txt
```

## Run

```powershell
.\scripts\run_server.ps1
```

Open:

```text
http://127.0.0.1:5000
```

Equivalent direct command:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m sensecv.app
```

## Data Configuration

The app defaults to these local paths:

```text
data/clips
data/datasets
data/exports
data/uploaded_datasets
data/history.json
```

Override them with environment variables if needed. See `.env.example` for the full list.

## Batch Scripts

```powershell
python scripts\export_sensecv.py sensecv walking
python scripts\run_sensecv_02062026_dronet_3fps.py
python scripts\run_samsung_dronet_samples.py
```

Most scripts accept environment overrides for dataset roots and output directories.

## Deploy

From the project root:

```powershell
flyctl deploy -c deploy\fly.toml
```

The container uses `src/sensecv/app.py` through `gunicorn sensecv.app:app` and stores production mutable data on the Fly volume mounted at `/data`.

## Documentation

Start with `docs/wiki/index.md` and `docs/wiki/overview.md`.