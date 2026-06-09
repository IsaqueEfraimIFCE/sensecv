# Inference Pipeline

Source: `dronet/run_dronet.py`.

## Inputs

- Weights: `dronet/repo/model/model_weights.h5`.
- Video root: `C:\Users\Isaque\Downloads\Supermercado Telefrango (Sem GPS)\PilotGuru\exports`.
- Output root: `dronet_results/`.

## Per-Video Flow

1. Recursively discover `.mp4` files under the video root.
2. Open each video with OpenCV.
3. For every decoded frame:
   - Apply [[preprocessing]].
   - Run the PyTorch [[model-architecture]].
   - Record steering, yaw degrees, and collision probability.
   - Draw overlay text, steering needle, and collision bar.
4. Write per-clip CSV and annotated MP4.
5. Append all frame rows into `all_frames.csv`.
6. Write aggregate `summary.json`.

## Output Columns

Per-frame CSVs contain:

- `frame`
- `time_s`
- `steering`
- `yaw_deg`
- `collision_prob`

`all_frames.csv` adds `clip` and `video`.

## Aggregate Metrics

For each readable clip, `summary.json` stores frame count, FPS, resolution, steering mean/min/max, collision mean/max, and fraction of frames with collision probability at least `0.5`.

