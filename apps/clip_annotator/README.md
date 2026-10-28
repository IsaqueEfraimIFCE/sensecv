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

## Live deployment (Drive mode, no volume)

Live at **https://sensecv-clip-annotator.fly.dev/** (Fly app `sensecv-clip-annotator`,
1 machine in `gru`, 512 MB, **no volume**; it stops when idle and wakes on the
first request). Everything lives on Google Drive, like the frame annotator:

| What | Drive |
|---|---|
| Videos | `SenseCV clip annotator/videos/<clip_id>.mp4` (folder `1wZuWm2uYqxhJsB5E9qdtXEyLIJXUbSJ5`): 1045 recordings, 23 collections of `Dataset - SenseCV`, 16.35 GB |
| Index | `index.sqlite` in the videos folder, downloaded at every start |
| Annotations | `SenseCV clip annotator/annotations/<annotator>/<clip>.json` + `_progress.json` (folder `1ee_CThXlaN5WFymQq7elkjMUJxmeeHwb`) |

The machine keeps only temporary files (`/tmp/clip_annotator`: the index and a
1 GB video cache), rebuilt from Drive after each start. The first open of a clip
after a start fetches it from Drive (measured: ~1.5 s extra for a typical clip;
the 216 s recording takes a few seconds more); later opens come from the cache.
Saves reach Drive about 1 s later (write-behind); on a normal stop the server
flushes them first (`kill_signal`/`kill_timeout`).

Annotator slots `annotator01`..`annotator10`; tokens and invite links live
outside git in `Desktop/SenseCV_clip_annotator/` (`annotators.json`,
`invite_links.txt`). Assign people to slots; don't rename slots (the slot name
is the annotation folder).

**Keep the Drive token alive.** `GOOGLE_AUTHORIZED_USER_JSON` is the full-Drive
token from `colab-hello-world/token_drive.json`. While the OAuth consent screen
of project `gdrive-505701` is in "Testing", Google revokes it after 7 days and
the app stops working (health 503 / sign-in errors). Publish the consent screen
("In production"), consent again, and set the secret from the new file.

**Add collections** (runs on the server, which downloads from and uploads to
Drive; only ~1 h access tokens leave the laptop): write `zips.json`
(`[{"id","name","size"}]` of the new `SenseCV-*.zip` files) and an
`access_token.json`, copy both to the machine, then run
`python -m clip_annotator.drive_import ...` followed by
`GOOGLE_ACCESS_TOKEN_FILE=... python -m clip_annotator.publish --dataset <raw> --index <drive index copy>`.
Both are resumable; a long run picks up a replaced `access_token.json` without a
restart. Give the machine scratch space (a temporary volume) and ~2 GB of memory
for the import, then remove them again. Restart the app afterwards so it
downloads the new index.

**Back up the annotations:** they are plain JSON files on Drive; download the
`annotations` folder.

**Revoke / add annotators:** edit `annotators.json`, then
`fly secrets import -a sensecv-clip-annotator` with a line
`ANNOTATOR_TOKENS_JSON=<one-line json>` on stdin (the machine restarts).
After restoring or deleting annotation files on Drive, restart the machine: the
server caches annotations in memory.

Deploy code changes from `apps/clip_annotator/`: `fly deploy --ha=false`
(flyctl resolves `--config` relative to the build directory, so run it there).

Volume mode (`ANNOTATOR_STORAGE=volume` plus a `[mounts]` section, videos from
`publish --bundle`) still works for a deployment without any Google account.

The Google-credential caveats of the frame annotator apply unchanged (service
accounts cannot write to a personal Drive; a "Testing" OAuth consent screen
expires refresh tokens after 7 days).

`drive.py` and `make_tokens.py` are copies of the frame annotator's, so each app
stays a self-contained deploy unit (its own Docker build context).

## Development

```
pip install -r requirements.txt pytest
PYTHONPATH=. pytest tests
```
