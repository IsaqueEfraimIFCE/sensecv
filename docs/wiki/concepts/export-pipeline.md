---
type: concept
tags: [workflow, ffmpeg, export]
code_refs: [app.py]
updated: 2026-05-30
---

# Export pipeline

What happens when the operator clicks **Exportar** -> `POST /api/crop` ->
`api_crop()` in [[app-backend]].

## Steps
1. **Validate and name.** Sanitize the name; compose the folder from the
   [[classification-taxonomy]]. Reject empty names (400) or collisions (409)
   before writing anything. Create `exports/<folder>/`.
2. **Cut the video.** `api_crop()` and batch export both call `_ffmpeg_cut()`.
   The source video is resolved through `clip_video_path(idx)`, because raw
   clips use `video.mp4` while exports use `<folder>.mp4`.
3. **Save sensor data.** `save_sensor_data()` filters `frames`,
   `accelerations`, and `rotations` to `[t_start, t_end]` and re-bases every
   `time_usec` (and frame `sensor_timestamp`) so the window starts at 0. Frame
   ids re-number from 0. Output matches the raw schema ([[clip-data-model]]).
   Failure here is swallowed; the video still exports.
4. **Save SSIM audit.** `ssim_frame_selection()` writes
   `ssim_selection.json`; `save_ssim_review_videos()` writes
   `ssim_review/all_frames.mp4`, `chosen_frames.mp4`, and
   `not_chosen_frames.mp4` so the retained-frame count can be checked visually.
   SSIM/audit-video failures are non-fatal.
5. **Record.** Append a row to [[history-json]] and return the new
   export state/next number.

## ffmpeg command shape
Exports are re-encoded instead of stream-copied:

```text
ffmpeg -y -ss <start> -noautorotate -i <source-mp4> -t <duration> \
  -map 0:v:0 -an -map_metadata 0 \
  -c:v libx264 -preset ultrafast -crf 23 -pix_fmt yuv420p \
  -movflags +faststart <exports/folder/folder.mp4>
```

Why:
- `-c copy` produced keyframe-aligned fragments; short lateral-deviation exports
  could decode as only a few frames even when `history.json` had a ~1 s window.
- Re-encoding makes the trimmed MP4 start at timestamp 0 and play correctly as a
  source clip when selected from `exports/`.
- `-noautorotate` plus `-map_metadata 0` preserves the phone rotation metadata
  instead of baking a rotation into pixels, keeping the existing frontend
  orientation logic consistent.

On ffmpeg failure, `api_crop()` removes the partial export folder and returns
500. Batch export removes the failed per-clip folder and records the failure in
`sources.csv`.

## Output
A self-contained folder under `exports/`; see [[exports-output]]. Because the
sensor JSON is rebased and re-numbered, each export is independently consumable
and is also selectable in the viewer as `exports/<folder>`.

## Repair note
On 2026-05-30, 24 history-backed exports were repaired by re-cutting them with
the re-encode path from their original source clips. Two folders without matching
`history.json` records (`06_obstaculo_esquerda_desvio_direita`, `erro4`) were
left as-is.

