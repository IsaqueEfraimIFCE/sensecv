---
type: entity
tags: [data-model, output]
code_refs: [app.py]
updated: 2026-05-30
---

# exports/ - Export output

Where finished annotations land. Each export is a folder named by the
[[classification-taxonomy]] (for example
`01_obstaculo_centro_desvio_direita/`) containing a trimmed clip and its
time-rebased sensor data.

## Contents of an export folder
| File | Source | Notes |
|---|---|---|
| `<folder>.mp4` | `_ffmpeg_cut()` H.264 cut of the source MP4 | Re-encoded with libx264 so short clips play correctly; rotation metadata preserved |
| `frames.json` | Filtered `frames` within `[start,end]` | `time_usec` and `sensor_timestamp` rebased so start -> 0; `frame_id` re-numbered from 0 |
| `accelerations.json` | Filtered accel samples in window | `time_usec` rebased |
| `rotations.json` | Filtered gyro samples in window | `time_usec` rebased |

The sensor JSON mirrors the raw [[clip-data-model]] schema, so exports are
drop-in compatible with anything that consumes raw clips. The viewer also scans
`exports/` as an input group: export clips appear as `exports/<folder>`.

## Relationship to history
Every successful export appends a record to [[history-json]]. A folder name
collision (`name_exists()`) is rejected with HTTP 409 before files are written.

`history.json` is pruned against actual export folders by `load_history()`, so
deleting an export folder removes the stale ledger entry on the next history or
export-state read.

## Numbering
The next prefilled export number comes from actual folders in `exports/`, not
from the maximum historical record. `get_next_number()` parses leading digits
from folder names and returns the first missing positive integer; if there are
no numbered export folders, the UI shows `01`.

## Note
`exports/` is separate from raw clip folders; originals are never touched.
Batch dataset exports (`SenseCV dataset/`, `Supermercado dataset/`) are separate
again and intentionally do not write to this curated folder.

