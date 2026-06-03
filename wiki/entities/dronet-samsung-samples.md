---
type: entity
tags: [dronet, inference, samsung, SenseCV]
code_refs: [run_samsung_dronet_samples.py]
updated: 2026-05-31
---

# DroNet Samsung random samples

`run_samsung_dronet_samples.py` runs the local DroNet PyTorch port from
`C:\Users\Isaque\Desktop\dronet` over randomly sampled frames from the Samsung
30-05-2026 SenseCV gimbal dataset.

## Inputs

- Video root:
  `C:\Users\Isaque\Downloads\SenseCV-30-05-2026-IFCE-Gimbal-Samsung\SenseCV-30-05-2026-IFCE-Gimbal-Samsung`
- Metadata CSV:
  `C:\Users\Isaque\Downloads\Coleta  IFCE - 30 de maio de 2026 - ANDRÃ‰ (Samsung) - PÃ¡gina1.csv`
- DroNet weights:
  `C:\Users\Isaque\Desktop\dronet\repo\model\model_weights.h5`

The CSV has 53 physical rows, but only 48 non-empty unique `ID` values
(`1`-`48`). These match the 48 Samsung clip folders.

## Output

Output root:
`dronet_samsung_random_samples/`

Layout:

```text
dronet_samsung_random_samples/
  all_samples.csv
  summary.json
  01/
    samples.csv
    sample_01_frame_000001.png
    ...
  ...
  48/
```

Each clip gets up to 20 deterministic random frame samples, seeded by
`20260530:<clip_id>`. The PNG files show DroNet's central crop with overlayed
steering, yaw, collision probability, and available CSV location metadata.

## Latest Run

Run date: 2026-05-31.

- Processed 48 Samsung clip folders.
- Wrote 960 sampled predictions.
- Every detected clip decoded successfully.
- Highest mean collision clips: `44` (0.640441), `46` (0.551409), `48`
  (0.479272), `12` (0.445524), `29` (0.396818).
- Most positive mean steering clips: `34` (0.135699), `40` (0.133205), `44`
  (0.106672), `42` (0.094694), `37` (0.079426).
- Most negative mean steering clips: `18` (-0.293457), `16` (-0.257877), `03`
  (-0.249556), `23` (-0.221129), `21` (-0.219429).

Interpretation caveat: DroNet was trained for outdoor navigation, so these
scores are model outputs for inspection, not validated labels for the indoor
IFCE phone-video dataset.

