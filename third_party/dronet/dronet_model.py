"""
DroNet (Loquercio et al., RA-L 2018) ResNet-8 reimplemented in PyTorch,
loading the original Keras `model_weights.h5` from the rpg_public_dronet repo.

Faithful to the original Keras graph:
  * TensorFlow 'SAME' padding (asymmetric for stride-2 convs) reproduced manually
  * channels-last Flatten ordering (H, W, C) to match the Dense kernels
  * BatchNormalization epsilon = 1e-3 (Keras 2.0 default)
Outputs per frame: steering angle s in [-1, 1] and collision probability p in [0, 1].
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py


def _tf_same_pad(size, k, s):
    out = math.ceil(size / s)
    pad = max((out - 1) * s + k - size, 0)
    return pad // 2, pad - pad // 2  # (before, after)


class Conv2dSame(nn.Module):
    """Conv2d with TensorFlow-style 'SAME' padding."""
    def __init__(self, cin, cout, k, stride):
        super().__init__()
        self.k, self.stride = k, stride
        self.conv = nn.Conv2d(cin, cout, k, stride=stride, padding=0, bias=True)

    def forward(self, x):
        ph0, ph1 = _tf_same_pad(x.shape[2], self.k, self.stride)
        pw0, pw1 = _tf_same_pad(x.shape[3], self.k, self.stride)
        x = F.pad(x, (pw0, pw1, ph0, ph1))  # (W_left, W_right, H_top, H_bottom)
        return self.conv(x)


class ResNet8(nn.Module):
    def __init__(self):
        super().__init__()
        bn = lambda c: nn.BatchNorm2d(c, eps=1e-3)
        self.conv1 = Conv2dSame(1, 32, 5, 2)               # conv2d_1
        self.pool = nn.MaxPool2d(3, stride=2, padding=0)   # max_pooling2d_1

        # Residual block 1
        self.bn1 = bn(32); self.conv2 = Conv2dSame(32, 32, 3, 2)   # bn_1, conv2d_2
        self.bn2 = bn(32); self.conv3 = Conv2dSame(32, 32, 3, 1)   # bn_2, conv2d_3
        self.conv4 = Conv2dSame(32, 32, 1, 2)                      # conv2d_4 (shortcut)

        # Residual block 2
        self.bn3 = bn(32); self.conv5 = Conv2dSame(32, 64, 3, 2)   # bn_3, conv2d_5
        self.bn4 = bn(64); self.conv6 = Conv2dSame(64, 64, 3, 1)   # bn_4, conv2d_6
        self.conv7 = Conv2dSame(32, 64, 1, 2)                      # conv2d_7 (shortcut)

        # Residual block 3
        self.bn5 = bn(64); self.conv8 = Conv2dSame(64, 128, 3, 2)  # bn_5, conv2d_8
        self.bn6 = bn(128); self.conv9 = Conv2dSame(128, 128, 3, 1)# bn_6, conv2d_9
        self.conv10 = Conv2dSame(64, 128, 1, 2)                    # conv2d_10 (shortcut)

        self.dense_steer = nn.Linear(6272, 1)   # dense_1
        self.dense_coll = nn.Linear(6272, 1)    # dense_2

    def forward(self, x):
        x1 = self.pool(self.conv1(x))

        x2 = F.relu(self.bn1(x1)); x2 = self.conv2(x2)
        x2 = F.relu(self.bn2(x2)); x2 = self.conv3(x2)
        x3 = self.conv4(x1) + x2

        x4 = F.relu(self.bn3(x3)); x4 = self.conv5(x4)
        x4 = F.relu(self.bn4(x4)); x4 = self.conv6(x4)
        x5 = self.conv7(x3) + x4

        x6 = F.relu(self.bn5(x5)); x6 = self.conv8(x6)
        x6 = F.relu(self.bn6(x6)); x6 = self.conv9(x6)
        x7 = self.conv10(x5) + x6

        # Keras Flatten is channels-last: (N, H, W, C) -> (N, H*W*C)
        f = x7.permute(0, 2, 3, 1).reshape(x7.shape[0], -1)
        f = F.relu(f)  # activation_7 (dropout is identity at inference)
        steer = self.dense_steer(f)
        coll = torch.sigmoid(self.dense_coll(f))
        return steer, coll


def _g(h5, layer, weight):
    return np.asarray(h5[f"{layer}/{layer}/{weight}:0"])


def load_dronet(weights_path):
    m = ResNet8()
    h = h5py.File(weights_path, "r")

    conv_map = {
        "conv1": "conv2d_1", "conv2": "conv2d_2", "conv3": "conv2d_3",
        "conv4": "conv2d_4", "conv5": "conv2d_5", "conv6": "conv2d_6",
        "conv7": "conv2d_7", "conv8": "conv2d_8", "conv9": "conv2d_9",
        "conv10": "conv2d_10",
    }
    for attr, kname in conv_map.items():
        kernel = _g(h, kname, "kernel")     # (kh, kw, cin, cout)
        bias = _g(h, kname, "bias")
        w = np.transpose(kernel, (3, 2, 0, 1))  # -> (cout, cin, kh, kw)
        conv = getattr(m, attr).conv
        conv.weight.data = torch.from_numpy(w.copy())
        conv.bias.data = torch.from_numpy(bias.copy())

    bn_map = {
        "bn1": "batch_normalization_1", "bn2": "batch_normalization_2",
        "bn3": "batch_normalization_3", "bn4": "batch_normalization_4",
        "bn5": "batch_normalization_5", "bn6": "batch_normalization_6",
    }
    for attr, kname in bn_map.items():
        bn = getattr(m, attr)
        bn.weight.data = torch.from_numpy(_g(h, kname, "gamma").copy())
        bn.bias.data = torch.from_numpy(_g(h, kname, "beta").copy())
        bn.running_mean.data = torch.from_numpy(_g(h, kname, "moving_mean").copy())
        bn.running_var.data = torch.from_numpy(_g(h, kname, "moving_variance").copy())

    # dense_1 = steering, dense_2 = collision; Keras kernel (in, out) -> Linear (out, in)
    for layer, lin in (("dense_1", m.dense_steer), ("dense_2", m.dense_coll)):
        kernel = _g(h, layer, "kernel")     # (6272, 1)
        bias = _g(h, layer, "bias")
        lin.weight.data = torch.from_numpy(kernel.T.copy())
        lin.bias.data = torch.from_numpy(bias.copy())

    h.close()
    m.eval()
    return m


# ---- preprocessing, identical to repo's img_utils.load_img + rescale 1/255 ----
import cv2

TARGET_SIZE = (320, 240)   # (width, height) passed to cv2.resize
CROP = 200                 # central-width / bottom-height crop


def central_image_crop(img, crop_w=CROP, crop_h=CROP):
    half_w = int(img.shape[1] / 2)
    return img[img.shape[0] - crop_h: img.shape[0],
               half_w - int(crop_w / 2): half_w + int(crop_w / 2)]


def preprocess_bgr(frame_bgr):
    """BGR frame (any size) -> model input tensor (1,1,200,200), float32 in [0,1]."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, TARGET_SIZE)          # -> (240, 320)
    crop = central_image_crop(gray)               # -> (200, 200)
    arr = crop.astype(np.float32) / 255.0
    t = torch.from_numpy(arr)[None, None, :, :]   # (1,1,200,200)
    return t, crop
