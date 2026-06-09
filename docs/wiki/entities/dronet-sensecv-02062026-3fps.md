---
type: entity
tags: [dronet, inference, SenseCV, gimbal]
code_refs: [run_sensecv_02062026_dronet_3fps.py]
updated: 2026-06-03
---

# DroNet SenseCV 02-06-2026 3 FPS

`run_sensecv_02062026_dronet_3fps.py` runs the local DroNet PyTorch port from
`PilotGuru\dronet` over every clip in the 02-06-2026 SenseCV
gimbal dataset, sampled at 3 frames per second.

This differs from [[dronet-samsung-samples]]: it is exhaustive over the target
folder at a fixed sample rate, not a deterministic random subset.

It also differs from [[dronet-live-classification]], which runs on-demand from
Flask for the viewer's current paused frame or 3 FPS playback bucket.

## Inputs

- Video root:
  `SenseCV-02-06-2026-IFCE-Gimbal\SenseCV-02-06-2026-IFCE-Gimbal`
- DroNet checkout:
  `PilotGuru\dronet`
- DroNet weights:
  `dronet\repo\model\model_weights.h5`

The dataset has 32 clip folders (`01` through `32`), each with `video.mp4`.

## Output

Output root:
`dronet_sensecv_02062026_3fps/`

Layout:

```text
dronet_sensecv_02062026_3fps/
  all_classifications.csv
  summary.json
  01/
    classifications.csv
    classification_001_frame_000000.png
    ...
    contact_sheet.png
  ...
  32/
```

Each annotated PNG shows the DroNet bottom-centered crop plus overlay text for
clip, sample number, source frame, sample time, steering, yaw, direction, and
collision probability. Each subfolder has its own `contact_sheet.png` containing
all generated images for that clip.

## Sampling

The script samples times at `0.0`, `0.3333`, `0.6667`, ... seconds until the
video duration, then maps each sample time to the nearest source frame:

```text
frame_idx = round(time_s * source_fps)
```

The videos are roughly 29.4-29.7 FPS at `720x720`, so most clips produce
18-25 DroNet classifications.

## Latest Run

Run date: 2026-06-03.

- Processed 32 clip folders.
- Wrote 666 frame classifications at 3 FPS.
- Wrote 32 per-clip `contact_sheet.png` files.
- Every detected clip decoded successfully.
- Highest mean collision clips: `21` (0.808611), `25` (0.787887), `26`
  (0.749860), `22` (0.740218), `32` (0.636719), `23` (0.587829), `24`
  (0.587827), `31` (0.580713).
- Largest steering extremes by absolute min/max: `15`, `24`, `23`, `09`,
  `13`, `10`, `17`, `31`.

Interpretation caveat: DroNet was trained for outdoor navigation, so these
scores are model outputs for visual inspection, not validated indoor obstacle
labels for the SenseCV phone-video dataset.
