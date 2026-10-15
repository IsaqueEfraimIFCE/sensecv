# SenseCV frame annotator

Standalone web app where external annotators label **individual frames** of the
SenseCV dataset (obstacle: free_path/obstacle, deviation: left/right/none, plus an
`unusable` flag). No existing labels are shown. Each annotator's work is stored
separately and can be resumed at any time.

## Where things live

| What | Where |
|---|---|
| Frames | Google Drive, as per-clip sprite sheets (`<Drive>/SenseCV frame annotator/sheets/<clip>__sNN.jpg`, 40 frames per sheet, 256 px tiles) |
| Index (links only) | `index.sqlite` in the same Drive folder; the server downloads it at boot. Tables `clips` and `sheets` hold Drive file ids, no labels |
| Annotations | `<Drive>/SenseCV frame annotator/annotations/<annotator>/<clip>.json` (run-length rows `[start, end, obstacle, deviation, unusable]`) and `_progress.json` per annotator |
| Web app | Flask on Fly.io, one gunicorn worker (single-writer cache) |

Sprite sheets are a transport format only; every frame is labelled individually.

## Runbook

1. **Publish** (once; resumable):
   `GOOGLE_AUTHORIZED_USER_FILE=<token.json> python -m frame_annotator.publish --dataset <SenseCV_location_split_fullres> --index index.sqlite`
   Prints `SHEETS_DRIVE_FOLDER_ID` and `ANNOTATIONS_DRIVE_FOLDER_ID`.
2. **Tokens**: `python -m frame_annotator.make_tokens ana bruno ... --out annotators.json`
   (never committed). It prints one invite fragment per annotator: `https://<app>/#token=<token>`.
3. **Secrets**:
   `fly secrets set ANNOTATOR_TOKENS_JSON="$(cat annotators.json)" SHEETS_DRIVE_FOLDER_ID=... ANNOTATIONS_DRIVE_FOLDER_ID=... GOOGLE_AUTHORIZED_USER_JSON="$(cat token.json)"`
4. **Deploy**: `fly volumes create annotator_data --size 3 --region gru`, then
   `fly deploy --config apps/frame_annotator/fly.toml --dockerfile apps/frame_annotator/Dockerfile apps/frame_annotator`.
5. Annotators open their invite link; the token is exchanged for an HttpOnly cookie.

## Google credentials: read this

* A **service account cannot create files in a personal Gmail Drive** (no storage
  quota, `storageQuotaExceeded`). Use it only with a **Shared Drive**.
* For a personal Drive use a **user OAuth token** (`GOOGLE_AUTHORIZED_USER_JSON`).
  Its refresh token **expires after 7 days while the OAuth consent screen is in
  "Testing"**; publish the consent screen ("In production") before a long campaign.

## Development

```
pip install -r requirements.txt -r requirements-publish.txt pytest
PYTHONPATH=. pytest tests
```
