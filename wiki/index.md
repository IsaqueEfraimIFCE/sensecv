---
type: meta
tags: [index, catalog]
updated: 2026-06-03
---

# Wiki Index

Catalog of every page. Start at [[overview]]. Maintenance rules in
`CLAUDE.md`. Timeline in [[log]].

## Meta
| Page | Summary |
|---|---|
| [[overview]] | What SenseCV is, the data flow, and the whole map |
| `CLAUDE.md` | Schema and maintenance guide for this wiki |
| [[log]] | Append-only timeline of ingests / changes / queries |

## Entities
| Page | Summary | Code |
|---|---|---|
| [[app-backend]] | The Flask server: clip discovery, grouping, data, suggestions, export | `app.py` |
| [[viewer-frontend]] | Single-page viewer/annotator UI, grouped selectors, charts, crop, playback | `templates/index.html` |
| [[clip-data-model]] | Raw/export clip folder schema: MP4 plus frames/accel/rotation JSON | - |
| [[api-routes]] | HTTP and JSON API surface | `app.py` |
| [[deployment-operations]] | Fly.io Docker deploy, volume-backed clips, health checks | `Dockerfile`, `fly.toml` |
| [[github-repository]] | Local Git/GitHub project state, remote URL, commits, and ignored data policy | `README.md`, `.gitignore` |
| [[exports-output]] | `exports/` folder structure, selectable export clips, numbering | `app.py` |
| [[history-json]] | Pruned `history.json` export ledger and record shape | `app.py` |
| [[SenseCV-dataset]] | Batch walking-only datasets with provenance CSVs | `export_SenseCV.py` |
| [[dronet-samsung-samples]] | DroNet random-frame inference over the Samsung 30-05-2026 SenseCV root | `run_samsung_dronet_samples.py` |
| [[dronet-sensecv-02062026-3fps]] | DroNet 3 FPS frame classifications and contact sheets for the 02-06-2026 SenseCV gimbal root | `run_sensecv_02062026_dronet_3fps.py` |
| [[dronet-live-classification]] | Live Flask/API DroNet inference shown in the viewer at paused frames or 3 FPS during playback | `app.py`, `templates/index.html` |

## Concepts
| Page | Summary | Code |
|---|---|---|
| [[velocity-estimation]] | GPS-free speed via IMU dead-reckoning and drift control | `app.py` |
| [[orientation-detection]] | Detecting the vertical/portrait usable footage state | `app.py` |
| [[walking-detection]] | Gait from acceleration-magnitude oscillation | `app.py` |
| [[crop-suggestion]] | `lateral` / `walking` / `vertical` auto-crop order and logic | `app.py` |
| [[classification-taxonomy]] | Obstacle/response label tree to folder names | both |
| [[export-pipeline]] | Re-encoded ffmpeg cut plus time-rebased sensor save and history | `app.py` |
| [[video-sensor-sync]] | Playing video with charts/cursor/timeline in lockstep | `templates/index.html` |

## Page count
- Meta: 3; Entities: 12; Concepts: 7 -> **22 pages**


