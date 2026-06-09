# Model Architecture

## Upstream Keras Model

Source: `dronet/repo/cnn_models.py`.

DroNet uses a forked CNN:

- Input shape: grayscale image, height x width x channels.
- Initial `Conv2D(32, 5x5, stride 2, same)` followed by `MaxPooling2D(3x3, stride 2)`.
- Three residual blocks:
  - Block 1 keeps 32 channels and downsamples with stride 2.
  - Block 2 increases to 64 channels and downsamples with stride 2.
  - Block 3 increases to 128 channels and downsamples with stride 2.
- Flatten, ReLU, dropout.
- Two dense heads:
  - Steering regression.
  - Collision probability with sigmoid activation.

## Local PyTorch Port

Source: `dronet/dronet_model.py`.

The local port implements `ResNet8` in PyTorch and loads the original Keras `model_weights.h5` directly with `h5py`.

Compatibility details:

- TensorFlow `SAME` padding is reproduced manually, including asymmetric padding for stride-2 convolutions.
- Keras kernels are transposed from `(kh, kw, cin, cout)` to PyTorch `(cout, cin, kh, kw)`.
- Keras dense kernels are transposed for PyTorch linear layers.
- Keras channels-last flatten order is preserved with `permute(0, 2, 3, 1)` before reshape.
- BatchNorm uses epsilon `1e-3`.
- Dropout is effectively omitted at inference because the loaded model is set to eval mode.

## Parameter Count

`dronet_results/README.md` records a parameter count of 320,930 for the local PyTorch port.

## Outputs

- `steering`: raw steering output, intended as approximately `[-1, 1]`.
- `collision_prob`: sigmoid collision probability in `[0, 1]`.

