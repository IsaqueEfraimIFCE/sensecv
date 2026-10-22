# SenseCV clip annotator

Standalone web app where annotators watch **whole SenseCV recordings** and mark
**when** each event happens and **what kind** it is. It is the full-clip sibling
of the per-frame [`frame_annotator`](../frame_annotator/README.md) (same sign-in,
storage and deploy design) and looks like the SenseCV viewer (`src/sensecv`):
same colour tokens, video + canvas timeline + frame-step bar, and the
crop-form chips (Desvio/Parada, Esquerda/Direita, obstacle position).

No existing label is shown. Each annotator's work is stored separately and can
be resumed at any time. The interface is in Portuguese.

## What gets annotated

Per clip, a list of non-overlapping events:

| Field | Values |
|---|---|
| `kind` | `desvio` (lateral deviation) or `parada` (stop) |
| `side` | `esquerda` / `direita` - where the person goes; required for a desvio, `null` for a parada |
| `obstacle` | `esquerda` / `centro` / `direita` / `nao_visivel`, optional |
| `start_s` (T1) | the body starts to react (turn away / brake) |
| `end_s` (T2) | trajectory stable again / fully stopped; required for a desvio, optional for a parada |
| `start_frame`, `end_frame` | `floor(t * fps)`, the frame shown at that instant (derived, as in the viewer) |

T1/T2 follow `criterios_imu_video_orientando` (T1 = physical start, T2 = return
to a stabilised trajectory). A clip is concluded as `eventos` (at least one
event), `sem_evento` (free walk) or `inutilizavel`, plus an optional note.

Stored file (`annotations/<annotator>/<clip>.json`):

```json
{"clip_id": "SenseCV-09-12-2026-SUPERMERCADO__01", "duration_s": 8.601, "fps": 30.0,
 "events": [{"kind": "desvio", "side": "direita", "obstacle": "centro",
             "start_s": 2.5, "end_s": 3.55, "start_frame": 75, "end_frame": 106}],
 "verdict": "eventos", "done": true, "note": "", "status": "concluido",
 "updated_at": "2026-10-22T14:15:17Z"}
```

Times are seconds of the video timeline. The duration comes from the MP4
header (`moov/mvhd`), not `metadata.json`, which reports ~0.7 s too much. To line
events up with `frames.json` / IMU, convert through time, not through frame ids.

## Where things live

| What | Where |
|---|---|
| Videos | Google Drive, `<Drive>/SenseCV clip annotator/videos/<clip_id>.mp4` (original phone MP4s, 720x480, a few MB each; no transcoding) |
| Index (links only) | `index.sqlite` in the same Drive folder; downloaded at boot. Table `clips`: id, collection, recording, duration, fps, frame count, Drive file id |
| Annotations | `<Drive>/SenseCV clip annotator/annotations/<annotator>/<clip>.json` and `_progress.json` per annotator |
| Web app | Flask on Fly.io, one gunicorn worker (single-writer cache); videos are cached on the volume and served with HTTP Range |

A full clip is a recording folder with an `.mp4`, `frames.json` and
`metadata.json`; exported cuts (`cut_info.json`) and `SenseCV-Teste-Estresse`
are skipped. Clip id = `<collection>__<recording>`.

## Try it locally (no Drive)

```
pip install -r requirements.txt
python -m clip_annotator.dev --dataset C:/Users/<you>/Desktop/SenseCV --data ./devdata
```

It indexes the recordings, keeps annotations in `./devdata/annotations`, and
prints an invite link (`http://127.0.0.1:5100/#token=...`). Add
`--annotator ana --annotator bruno` for more people.

## Runbook (production)

1. **Publish** (once; resumable; upload only, no heavy processing):
   `GOOGLE_AUTHORIZED_USER_FILE=<token.json> python -m clip_annotator.publish --dataset <recordings root> --index index.sqlite`
   Prints `VIDEOS_DRIVE_FOLDER_ID` and `ANNOTATIONS_DRIVE_FOLDER_ID`.
2. **Tokens**: `python -m clip_annotator.make_tokens ana bruno ... --out annotators.json`
   (never committed). Invite links are `https://<app>/#token=<token>`.
3. **Secrets**:
   `fly secrets set ANNOTATOR_TOKENS_JSON="$(cat annotators.json)" VIDEOS_DRIVE_FOLDER_ID=... ANNOTATIONS_DRIVE_FOLDER_ID=... GOOGLE_AUTHORIZED_USER_JSON="$(cat token.json)"`
4. **Deploy**: `fly volumes create clip_annotator_data --size 3 --region gru`, then
   `fly deploy --config apps/clip_annotator/fly.toml --dockerfile apps/clip_annotator/Dockerfile apps/clip_annotator`.

The Google-credential caveats of the frame annotator apply unchanged (service
accounts cannot write to a personal Drive; a "Testing" OAuth consent screen
expires refresh tokens after 7 days). After restoring or deleting annotation
files on Drive, restart the machine: the server caches annotations in memory.

`drive.py` and `make_tokens.py` are copies of the frame annotator's, so each app
stays a self-contained deploy unit (its own Docker build context).

## Development

```
pip install -r requirements.txt pytest
PYTHONPATH=. pytest tests
```
