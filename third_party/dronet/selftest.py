import os

import numpy as np, torch, cv2
from dronet_model import load_dronet, preprocess_bgr

DRONET_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(DRONET_DIR))
W = os.environ.get("DRONET_WEIGHTS", os.path.join(DRONET_DIR, "repo", "model", "model_weights.h5"))
m = load_dronet(W)
n_params = sum(p.numel() for p in m.parameters())
print("loaded ok, params:", n_params)

# shape check with a dummy frame
dummy = (np.random.rand(3840, 2160, 3) * 255).astype(np.uint8)
t, crop = preprocess_bgr(dummy)
print("input tensor:", tuple(t.shape), "crop:", crop.shape)
with torch.no_grad():
    s, c = m(t)
print("random-frame  steer=%.4f  coll=%.4f" % (s.item(), c.item()))

# constant images -> sanity of ranges
for val in (0.0, 0.5, 1.0):
    img = np.full((3840, 2160, 3), int(val * 255), np.uint8)
    t, _ = preprocess_bgr(img)
    with torch.no_grad():
        s, c = m(t)
    print("const %.1f      steer=%.4f  coll=%.4f" % (val, s.item(), c.item()))

# pull a real frame from a working video
sample_video = os.environ.get(
    "DRONET_SELFTEST_VIDEO",
    os.path.join(PROJECT_ROOT, "exports", "03", "008_sem_obstaculos.mp4"),
)
cap = cv2.VideoCapture(sample_video)
ok, fr = cap.read(); cap.release()
print("real frame read:", ok, None if not ok else fr.shape)
if ok:
    t, _ = preprocess_bgr(fr)
    with torch.no_grad():
        s, c = m(t)
    print("real-frame    steer=%.4f (yaw %.1f deg)  coll=%.4f" % (s.item(), s.item()*90, c.item()))
