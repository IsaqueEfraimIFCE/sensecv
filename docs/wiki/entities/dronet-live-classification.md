---
type: entity
tags: [dronet, inference, backend, frontend]
code_refs: [app.py, templates/index.html]
updated: 2026-06-03
---

# DroNet live classification

SenseCV can run DroNet inference from the Flask app for the frame currently
shown in the viewer. This is separate from the offline batch sheets in
[[dronet-sensecv-02062026-3fps]]: live classification is on-demand, driven by
the browser's current playback time.

## Backend

`app.py` defines:

- `DRONET_DIR`, defaulting to `PilotGuru\dronet`.
- `DRONET_WEIGHTS`, defaulting to
  `dronet\repo\model\model_weights.h5`.
- `DRONET_SAMPLE_FPS = 3.0`.
- `_load_dronet_runtime()`, which lazily imports `torch`,
  `load_dronet()`, and `preprocess_bgr()` only on the first live inference
  request.
- `dronet_frame_classification(idx, time_s, exact=False)`, which opens the
  clip video with OpenCV, maps requested time to a source frame, preprocesses
  that BGR frame with the local DroNet crop contract, and returns steering /
  yaw / collision fields.

The lazy load keeps normal clip browsing fast: opening the app and calling
`/api/data/<idx>` does not load the neural network. The first DroNet request
loads the H5 weights and caches the runtime in `_dronet_runtime`.

## API

Route:

```text
GET /api/dronet/<idx>?time=<seconds>&exact=<0|1>
```

`idx` is the current clip index from `CLIPS`. `time` is the browser video time
in seconds.

When `exact=1`, the backend maps `time` directly to the nearest video frame:

```text
frame_idx = round(time_s * source_fps)
```

When `exact=0`, the backend first buckets the requested time to a 3 FPS cadence:

```text
sample_time = floor(time_s * 3.0) / 3.0
frame_idx = round(sample_time * source_fps)
```

Successful response:

```jsonc
{
  "available": true,
  "clip": "SenseCV-02-06-2026-IFCE-Gimbal/01",
  "frame": 30,
  "time_s": 0.9993,
  "requested_time_s": 1.23,
  "sample_fps": 3.0,
  "exact": false,
  "source_fps": 30.0213,
  "steering": 0.0577,
  "yaw_deg": 5.1956,
  "direction": "STRAIGHT",
  "collision_prob": 0.9995,
  "collision_label": "COLLISION"
}
```

Unavailable/error responses return `available:false` and `error`. The route
uses HTTP 503 for runtime/video/inference failures and 404 for out-of-range
clip indexes.

## Cache

`_dronet_cache` is an `OrderedDict` capped at 300 entries. The key is:

```text
(clip display name, video mtime, frame_idx)
```

The video modification time invalidates stale cached predictions if the source
MP4 changes. The frame key means paused-frame queries and 3 FPS playback
queries share cached results when they land on the same decoded frame.

## Frontend

`templates/index.html` adds a compact **DroNet** panel under the live sensor
value boxes:

- `Direcao`: direction plus steering value.
- `Yaw`: steering converted to degrees (`steering * 90`).
- `Colisao`: collision probability, with red border when probability is at
  least `0.5`.
- Status line: source frame, inference time, mode (`frame pausado` or `3 fps`),
  and collision label.

The relevant JavaScript state/functions are:

- `resetDronetPanel()` clears stale predictions on `loadClip()`.
- `requestDronet(exact=false)` deduplicates requests and serializes in-flight
  inference so the UI does not stack multiple slow model calls.
- `renderDronet()` writes the returned direction/yaw/collision fields into the
  panel and shows errors when the backend reports `available:false`.

Update policy:

- On clip load, the UI requests `exact=1` for the initial paused frame.
- While playing, `syncLoop()` calls `requestDronet(false)`. The frontend
  dedupe key is `floor(currentTime * 3)`, so the backend is hit at most once
  per 3 FPS bucket.
- When paused or scrubbed while paused, `onTime()` and the `pause` event call
  `requestDronet(true)`, so the panel tracks the exact paused frame.

## Verification

Latest local verification:

- `python -m py_compile app.py`
- Flask test client:
  `/api/dronet/0?time=0&exact=1`
- Live server on alternate port because port 5000 had a stale/nonmatching
  process:
  `http://127.0.0.1:5001`
- Verified:
  `/health` returned `{"clips":313,"status":"ok"}`.
- Verified:
  `/api/dronet/0?time=1.23` returned a 3 FPS bucketed classification for frame
  `30`.

Interpretation caveat: DroNet outputs are inspection signals from an outdoor
navigation model, not validated indoor SenseCV labels.
