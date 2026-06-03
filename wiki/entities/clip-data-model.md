---
type: entity
tags: [data-model, sensors, raw-source]
code_refs: [app.py]
updated: 2026-06-02
---

# Clip data model

A clip is a folder with frame timing JSON, accelerometer JSON, gyroscope JSON,
and an MP4 video. Raw source clips normally live in timestamped folders such as
`2026_01_16-12_44_51/` and use `video.mp4`. Exported clips live under
`exports/<folder>/` and use `<folder>.mp4`.

`find_clips()` accepts either video naming pattern through `_clip_video_file()`;
the backend stores the actual MP4 path in `CLIP_VIDEO_PATHS`.

## Files in a clip folder

| File | Shape | Notes |
|---|---|---|
| `video.mp4` or `<folder>.mp4` | H.264 video | Served with byte-range support via `/video/<idx>` |
| `frames.json` | `{ "frames": [ {frame_id, time_usec, sensor_timestamp, ...} ] }` | Per-video-frame timing; `time_usec` is the master clock |
| `accelerations.json` | `{ "accelerations": [ {time_usec, x, y, z} ] }` | Raw accelerometer including gravity (m/s^2) |
| `rotations.json` | `{ "rotations": [ {time_usec, x, y, z} ] }` | Gyroscope angular velocity (rad/s) |
| `external_sensors.json` | `{ "external_sensors": [ {time_usec, button} ] }` | Optional external input stream; `button: 1` marks active input |

## Timebase
- All streams share a microsecond clock `time_usec`.
- `t0 = frames[0].time_usec`; the UI/charts use seconds relative to `t0`.
- Sensor sample rate is higher than frame rate, so the backend interpolates or
  nearest-samples sensors onto frame times.
- Optional external input is sampled as a step signal: each video frame uses the
  latest preceding `external_sensors` sample.
- `fps` is derived from the median inter-frame gap.

## Source groups
The UI uses `CLIP_GROUPS` to group the selector:
- `Supermercado Telefrango` for primary-root raw clips.
- One group per SenseCV extra root.
- `exports` for exported clips.

## Axis convention
Held in portrait, screen toward the user: `+y` is approximately up, and `z` is
out of the screen. Vertical detection thresholds rely on this convention.

## Immutability
Raw clip folders are read-only source of truth. Exports are rewritten only when
the user exports, deletes, or deliberately repairs export outputs.

