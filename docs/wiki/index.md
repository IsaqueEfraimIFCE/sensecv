---
type: meta
tags: [index, catalog]
updated: 2026-06-19
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
| [[source-inventory]] | DroNet source layer, upstream repo, model artifacts, and generated-result inventory |
| [[open-questions]] | DroNet caveats, unresolved validation questions, and follow-up checks |

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
| [[clear-dataset]] | Clear/no-obstacle image pool, SSIM supplement repair, DINOv2/LPIPS/VIF variants, and Kaggle clear source | clear dataset scripts |
| [[sensecv-02062026-ifce-clip-manifest]] | Spreadsheet manifest for the 32 uploaded IFCE 02-06-2026 clips: object, phone pose, obstacle side, deviation side, and height | - |
| [[dronet-samsung-samples]] | DroNet random-frame inference over the Samsung 30-05-2026 SenseCV root | `run_samsung_dronet_samples.py` |
| [[dronet-sensecv-02062026-3fps]] | DroNet 3 FPS frame classifications and contact sheets for the 02-06-2026 SenseCV gimbal root | `run_sensecv_02062026_dronet_3fps.py` |
| [[dronet-live-classification]] | Live Flask/API DroNet inference shown in the viewer at paused frames or 3 FPS during playback | `app.py`, `templates/index.html` |
| [[kaggle-two-head-dataset]] | Kaggle obstacle/deviation two-head image dataset built from manifest-export lateral cuts plus the clear set | `scripts/make_kaggle_two_head_dataset.py` |

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
| [[dataset-normalization]] | Rebuilding clip folders at 14.730 FPS video and 420 Hz sensors | `normalize_dataset_rates.py` |
| [[activation-maps]] | Grad-CAM heatmaps for the DroNet collision head and the SenseCV two-head obstacle/deviation heads, shown in the Ativação tab | `src/sensecv/app.py`, `src/sensecv/templates/index.html` |
| [[dronet-overview]] | DroNet research/runtime source layer vendored into SenseCV | `third_party/dronet/` |
| [[dronet-paper]] | Loquercio et al. DroNet paper claims, scope, and caveats | `third_party/dronet/RAL18_Loquercio.pdf` |
| [[model-architecture]] | Upstream Keras ResNet-8 and local PyTorch compatibility port | `third_party/dronet/dronet_model.py` |
| [[preprocessing]] | Original grayscale resize, bottom-center crop, and normalization contract | `third_party/dronet/dronet_model.py`, `third_party/dronet/repo/img_utils.py` |
| [[inference-pipeline]] | Batch DroNet inference over exports and output schema | `third_party/dronet/run_dronet.py` |
| [[experiment-results]] | Historical DroNet result summaries over supermarket export clips | `data/dronet_results/` |
| [[pilotguru-10fps-folder-check]] | 10 FPS folder-label expectation check against DroNet output thresholds | `data/dronet_results/pilotguru_10fps_match/` |
| [[ros-control]] | Upstream ROS/Bebop perception and control packages | `third_party/dronet/repo/drone_control/` |
| [[imu-event-labeling]] | IMU capture validation, T1/T2 event detection, and decision-window labeling from the criterios PDF | `app.py`, `scripts/imu_event_report.py` |

## Page count
- Meta: 6; Entities: 15; Concepts: 18 -> **39 pages**
