---
type: concept
tags: [dronet, sensemodel, gradcam, interpretability, backend, frontend]
code_refs: [src/sensecv/app.py, src/sensecv/templates/index.html, third_party/dronet/dronet_model.py]
updated: 2026-06-19
---

# Activation maps (Grad-CAM)

The viewer's **Ativação** tab shows where each model "looks" in the frame
currently in the player, as **Grad-CAM** heatmaps blended over the exact pixels
that model received. Three maps are rendered side by side:

| Map | Model | Head | Labels |
|---|---|---|---|
| DroNet — colisão | DroNet ResNet-8 (PyTorch port) | collision / classification | `COLLISION` / `CLEAR` |
| SenseCV — obstáculo | SenseCV two-head `.keras` | head 1 (obstacle) | `NONE` / `OBSTACLE` |
| SenseCV — desvio | SenseCV two-head `.keras` | head 2 (deviation) | `LEFT` / `RIGHT` / `NONE` |

See [[model-architecture]] for the DroNet graph and
[[dronet-live-classification]] for the sibling numeric panels these maps mirror.

## What Grad-CAM computes

For a chosen scalar output (here the predicted-class score / collision output),
Grad-CAM weights the last convolutional feature maps by the mean of the output's
gradient w.r.t. each channel, sums them, applies ReLU, and upsamples the result
to a coarse class-discriminative heatmap. High values are the regions that most
raised that prediction.

## Backend

`app.py › activation_maps_frame(idx, time_s, exact)` decodes the requested frame
**once** (same 3-fps-bucket / exact-frame mapping and `(clip, mtime, frame)`
caching as `dronet_frame_classification` / `sensemodel_frame_classification`,
see [[dronet-live-classification]]) and builds all three maps. Results are cached
in `_activation_cache` (24 entries).

- `_dronet_activation(frame)` — registers a forward + full-backward hook on
  `model.conv9.conv` (the last 3×3 conv of residual block 3, see
  [[model-architecture]]), runs the model with grad enabled, backprops the
  collision output, and forms the CAM from `relu(Σ_c mean(grad_c)·act_c)`. The
  base image is the 200×200 grayscale DroNet crop (see [[preprocessing]]).
- `_sensemodel_activation(frame)` — builds a `tf.keras.Model` mapping the input
  to `[last_conv.output] + model.outputs`, then a single persistent
  `GradientTape` yields **one CAM per head**. The last conv is found by
  `_keras_last_conv_layer()` (last layer with a 4D output). Heads are matched by
  output width (2 → obstacle, 3 → deviation), exactly like
  `_split_head_arrays` in [[api-routes]]. The base image is the raw 0–255
  RGB model input (no `/255`; the model has the embedded `Rescaling` layer).
- `_cam_to_data_uri(cam, base_bgr)` — min-max normalizes the CAM, applies
  `COLORMAP_JET`, `addWeighted`-blends it 50/50 over the base, PNG-encodes, and
  returns a `data:image/png;base64,…` URI.

Each sub-map degrades independently: a missing PyTorch or TensorFlow runtime
returns `{available:false, error}` for just that map while the others still
render.

### Route
`GET /api/activation/<idx>?time=<s>&exact=<0|1>` (see [[api-routes]]):

```jsonc
{ "available": true, "frame": 30, "time_s": 0.999, "exact": true,
  "model": "best_model.keras",
  "dronet":    { "available": true, "label": "CLEAR",    "prob": 0.0002, "image": "data:image/png;base64,..." },
  "obstacle":  { "available": true, "label": "OBSTACLE", "prob": 0.985,  "image": "data:image/png;base64,..." },
  "deviation": { "available": true, "label": "LEFT",     "prob": 0.970,  "image": "data:image/png;base64,..." } }
```

503 on a whole-frame failure (video/decoupled), 404 for an out-of-range index.

## Frontend

The `ativacao` view (in [[viewer-frontend]]) is **video-centric** — the player
stays mounted so the operator can scrub. `requestActivation(exact)` is called
from inside `requestDronet()` (sharing the play→3 fps / paused→exact triggers)
but **only fetches when the tab is active** (`activationActive()`), because each
request runs two backprops. `renderActivation()` writes the three PNGs and
label/probability lines; `resetActivationPanel()` clears them on clip load (via
`resetSensemodelPanel`). Entering the tab (`setView('ativacao')`) requests the
exact current frame.

## Caveats

- The DroNet maps are **inspection signals from an outdoor navigation model**,
  not validated indoor SenseCV labels (same caveat as
  [[dronet-live-classification]]).
- Grad-CAM resolution is the last conv map's spatial size, so the heatmaps are
  coarse by construction.
