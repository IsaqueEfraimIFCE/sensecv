# Preprocessing

Sources: `dronet/repo/img_utils.py`, `dronet/dronet_model.py`, `dronet_results/README.md`.

## Original Contract

The upstream data path loads an image, optionally converts to grayscale, resizes, applies a central-width and bottom-height crop, reshapes grayscale images with a channel dimension, and rescales by `1/255` in the generator.

The crop is not centered vertically. It takes the bottom `crop_height` pixels and the center `crop_width` pixels.

## Local Inference Contract

`preprocess_bgr(frame_bgr)` performs:

1. BGR frame to grayscale.
2. Resize to `320x240` as `(width, height)` in OpenCV.
3. Crop to the bottom-centered `200x200` region.
4. Convert to `float32` and divide by `255`.
5. Return tensor shape `(1, 1, 200, 200)`.

## Why It Matters

The PyTorch port is intended to be weight-compatible with the original Keras model. Small deviations in resize, crop, channel order, scaling, flatten order, or padding would change predictions and invalidate comparison with the upstream model.

## Local Domain Issue

The local videos are portrait phone videos. The preprocessing resizes them to `320x240` before taking the `200x200` bottom crop. That can distort scene geometry and remove context. This is a likely contributor to high collision probabilities in [[experiment-results]].

