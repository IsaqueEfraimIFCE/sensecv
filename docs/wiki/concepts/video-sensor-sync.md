---
type: concept
tags: [frontend, playback, sync]
code_refs: [templates/index.html]
updated: 2026-05-26
---

# Videoâ€“sensor synchronization

How the [[viewer-frontend]] keeps the video, the three charts, the live value
boxes, and the timeline playhead in lockstep so you can *watch the sensors move
with the footage*.

## Frame mapping
`nearestIdx(t)` binary-searches `DATA.times` for the frame nearest the video's
`currentTime`. All per-frame arrays (`accel`, `rotation`, `velocity`) are indexed
by that frame, so a video time maps directly to sensor values.

## `onTime()` â€” the update hub
On every tick it: updates the time/frame labels, sets each chart's cursor
(`_cursorX`) via the `vline` plugin, refreshes the value boxes, and redraws the
timeline playhead. Fired from `timeupdate` and `seeking` events (covers scrubbing
and arrow-key seeks).

## Smooth playback loop
`timeupdate` only fires ~4Ã—/s, which makes the cursor stutter during playback.
So while playing, a `requestAnimationFrame` loop (`syncLoop()`) calls `onTime()`
every animation frame for a smooth cursor. It starts on `play`, stops on
`pause`/`ended`. The play/pause button text and `Space` shortcut are wired to the
same `<video>` element.

## Timeline canvas
Drawn in `drawTimeline()`: the purple **speed waveform** (from
[[velocity-estimation]]), the blue crop region with green **S** / red **E**
markers, and the gold playhead. Click or drag on it to seek (`px2t`).

## Crop overlay on charts
The `cropregion` Chart.js plugin shades `[cropStart, cropEnd]` and draws dashed
start/end lines across every chart, so the suggested/edited window
([[crop-suggestion]]) is visible against the sensor traces, not just the video.

