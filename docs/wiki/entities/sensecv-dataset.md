---
type: entity
tags: [dataset, export, SenseCV, supermercado, batch]
code_refs: [export_SenseCV.py, app.py]
updated: 2026-05-31
---

# Batch auto-cut datasets

Unlabeled, auto-suggested dataset folders that sit alongside the operator's
interactive `exports/`. Produced by either `export_SenseCV.py` (CLI) or
`POST /api/batch-export` (UI's "Exportação em lote" section,
[[viewer-frontend]]). All paths run through `app.py` `export_set()`.

Two **source presets** × two **cut modes** = up to four output folders:

| preset | source set | walking mode → folder | lateral mode → folder |
|---|---|---|---|
| `SenseCV` | extra `SenseCV-*` roots (`WALKING_ONLY`), horizontal | `SenseCV dataset/` | `SenseCV dataset (lateral)/` |
| `supermarket` | primary `CLIPS_DIR`, vertical supermercado clips | `Supermercado dataset/` | `Supermercado dataset (lateral)/` |

Recent run counts: SenseCV walking 103/103 ok · Supermercado walking 71 ok / 86
no-walking · SenseCV lateral 102 ok / 1 below threshold.

The app also ingests two 30-05-2026 gimbal source roots from Downloads:
Samsung (48 clips) and Poco (54 clips). These are input roots, not batch-output
folders. The Samsung metadata CSV was used by [[dronet-samsung-samples]] for
per-ID DroNet sample inference.

Cut modes (full detail in [[crop-suggestion]]):
- **walking** — `suggest_crop(idx, 'walking')`, the walking-only segment.
- **lateral** — `suggest_lateral_deviation(idx)`, the single longest sustained
  lateral-acceleration burst ≥ 0.2 s.

All four are deliberately **separate from `exports/`** (the classified, labeled
dataset that feeds [[history-json]] and trains the walking-window classifier).

## Why they're separate from `exports/`
- `exports/` is the **labeled** dataset — each folder name encodes
  classification ([[classification-taxonomy]]) and each entry is appended to
  [[history-json]]. It also acts as ground truth for the walking-window
  classifier ([[crop-suggestion]]).
- The batch datasets are **provenance-only** cuts: no classification, no
  `history.json` write, no learned-window feedback loop. Bypassing
  `history.json` keeps the two regimes (horizontal SenseCV vs. vertical
  supermercado) from cross-contaminating the classifier
  ([[orientation-detection]]).

## Layout
```
<dataset>/
  sources.csv
  <safe-name>/
    <safe-name>.mp4
    frames.json
    accelerations.json
    rotations.json
    ssim_selection.json
    ssim_review/
      all_frames.mp4
      chosen_frames.mp4
      not_chosen_frames.mp4
      manifest.json
  …
```
Folder/file naming: the clip's display name with `/` replaced by `_` so it is a
single legal filename. The mp4 keeps the same base name as the folder, matching
the convention `api_crop()` uses in `exports/`. Examples:
`SenseCV-23-05-2023-IFCE_06/` (SenseCV) and `2026_01_16-12_44_51/`
(supermercado — bare folder name, since primary-root clips have no prefix).

## `sources.csv` (provenance ledger)
One row per clip in the preset, including skips. Columns:

| column | meaning |
|---|---|
| `output_folder` | The per-clip subfolder name (empty subfolders are not created for skips) |
| `source_display` | Display name in [[app-backend]]'s `CLIPS` (e.g. `SenseCV-26-05-2026-IFCE/01`) |
| `source_root` | Root label — the basename of the input root in `EXTRA_CLIP_ROOTS`, or `supermercado` for the primary root |
| `source_subfolder` | The subfolder inside that root |
| `source_path` | Absolute path of the source clip folder |
| `mode` | `walking` or `lateral` — which detector produced the cut |
| `start`, `end`, `duration` | Crop window in seconds (relative to source clip start) |
| `frames_before`, `frames_after` | Source frames inside the cut and visually distinct images selected by SSIM |
| `ssim_threshold`, `ssim_status` | SSIM de-duplication threshold and status/error |
| `status` | `ok` · `ok_no_sensor: <err>` · `no_segment: <reason>` · `ffmpeg_error: <err>` |

## Pipeline (`app.export_set`)
The CLI (`export_SenseCV.py`) and the Flask route (`/api/batch-export`,
[[api-routes]]) are both thin wrappers around `app.py` `export_set(indices,
out_dir, mode)`. They share state — `CLIPS`, `CLIP_PATHS`, `WALKING_ONLY`,
`suggest_crop`, `suggest_lateral_deviation`, `save_sensor_data` — by living in
the same module rather than re-importing it.

1. Filter clips via `_preset_filter(preset)` — `name in WALKING_ONLY` for
   SenseCV, the complement for supermercado.
2. Per clip: call `suggest_crop(idx, 'walking')` when `mode='walking'` or
   `suggest_lateral_deviation(idx)` when `mode='lateral'` ([[crop-suggestion]]).
   Walking mode for supermercado goes through the learned-window / Random
   Forest classifier path gated on [[orientation-detection]]; walking for
   SenseCV uses the dedicated horizontal branch.
3. If found, `_ffmpeg_cut()` re-encodes a clean H.264 cut into
   `<dataset>/<name>/<name>.mp4`, preserving rotation metadata.
4. `save_sensor_data(idx, start, end, out_folder)` re-bases the per-frame and
   sensor timestamps to `0` at the cut start (same logic as the interactive
   export — see [[export-pipeline]]).
5. `ssim_frame_selection()` writes `ssim_selection.json` with the selected
   source frame IDs/times and records before/after counts in `sources.csv`.
   `save_ssim_review_videos()` also writes three audit videos under
   `ssim_review/`: all candidate frames in the crop, only the SSIM-retained
   frames, and the rejected/not-chosen frames. These are generated from the same
   frame records used by the selector, so the visual split matches the
   before/after counts.
6. Append a row to `sources.csv`.

Clips that don't yield a cut (`found:false`) are recorded in the CSV with
status `no_segment: <reason>` and no subfolder is created.

## Idempotency / gotchas
- Re-running either driver removes a previous `<name>/` subfolder before
  re-cutting it, so it is safe to re-run. The CSV is rewritten in full each
  run; it is not appended.
- `/api/batch-export` is **synchronous** — the HTTP request blocks until every
  clip has been processed and only then returns the summary. The UI handler
  disables all batch buttons while a run is in flight.
- A sensor-save failure is non-fatal (matches `api_crop()` behavior) — the
  video still ships and the row records `ok_no_sensor: …`.
- `learned_walking_window()` looks at `history.json`, which this script does
  **not** write. So re-running these batches never affects future suggestion
  behavior, and the two regimes do not bleed into each other. Note that the
  supermercado preset *does* read `history.json` indirectly: the 5 clips that
  exist in [[history-json]] use their learned window verbatim, and unexported
  ones use the Random Forest fallback trained on those same 5 windows.

