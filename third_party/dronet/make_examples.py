"""Produce 20 example tiles: the 200x200 image DroNet actually sees + its classification."""
import os, glob, math
import cv2, numpy as np, torch
from dronet_model import load_dronet, preprocess_bgr

DRONET_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(DRONET_DIR))
WEIGHTS = os.environ.get("DRONET_WEIGHTS", os.path.join(DRONET_DIR, "repo", "model", "model_weights.h5"))
BASE = os.environ.get("PILOTGURU_EXPORTS_DIR", os.path.join(PROJECT_ROOT, "data", "exports"))
OUT = os.environ.get("DRONET_EXAMPLES_DIR", os.path.join(PROJECT_ROOT, "data", "dronet_results", "examples"))
os.makedirs(OUT, exist_ok=True)

model = load_dronet(WEIGHTS)

# Build the full (clip, video_path, frame_idx) index across all readable clips.
index = []
for vpath in sorted(glob.glob(os.path.join(BASE, "**", "*.mp4"), recursive=True)):
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        cap.release(); continue
    n = 0
    while True:
        if not cap.grab(): break
        n += 1
    cap.release()
    folder = os.path.basename(os.path.dirname(vpath))
    for i in range(n):
        index.append((folder, vpath, i))

# 20 evenly-spaced samples across everything.
N = 20
picks = [index[round(k * (len(index) - 1) / (N - 1))] for k in range(N)]
print(f"{len(index)} frames available; sampling {N}")

def grab_frame(vpath, idx):
    cap = cv2.VideoCapture(vpath)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    if not ok:  # fallback: sequential
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(idx + 1):
            ok, fr = cap.read()
            if not ok: break
    cap.release()
    return fr

TILE_W, IMG = 240, 200          # tile width, image size
PANEL = 86                      # text panel height
tiles = []
for n, (folder, vpath, idx) in enumerate(picks):
    fr = grab_frame(vpath, idx)
    if fr is None:
        continue
    t, crop = preprocess_bgr(fr)          # crop = 200x200 grayscale (what model sees)
    with torch.no_grad():
        s, c = model(t)
    s, c = float(s.item()), float(c.item())
    yaw = s * 90.0
    direction = "STRAIGHT" if abs(s) < 0.1 else ("RIGHT" if s > 0 else "LEFT")
    coll_label = "COLLISION" if c >= 0.5 else "CLEAR"

    # tile = 200x200 input image (centered in TILE_W) + text panel
    img = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    pad = (TILE_W - IMG) // 2
    img = cv2.copyMakeBorder(img, 0, 0, pad, TILE_W - IMG - pad,
                             cv2.BORDER_CONSTANT, value=(40, 40, 40))
    panel = np.full((PANEL, TILE_W, 3), 25, np.uint8)
    cv2.putText(panel, f"#{n+1}  clip {folder.split('_')[0]}  f{idx}", (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(panel, f"steer {s:+.2f} ({yaw:+.0f}d) {direction}", (6, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
    ccol = (0, 0, 255) if c >= 0.5 else (0, 220, 220)
    cv2.putText(panel, f"coll {c:.2f}  {coll_label}", (6, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, ccol, 1, cv2.LINE_AA)
    # steering needle in panel
    cx, cy = TILE_W - 30, 60
    ex = int(cx + math.sin(math.radians(yaw)) * 18)
    ey = int(cy - math.cos(math.radians(yaw)) * 18)
    cv2.line(panel, (cx, cy), (ex, ey), (0, 255, 0), 2, cv2.LINE_AA)

    tile = np.vstack([img, panel])
    cv2.imwrite(os.path.join(OUT, f"example_{n+1:02d}_clip{folder.split('_')[0]}_f{idx}.png"), tile)
    tiles.append(tile)

# Contact sheet: 5 columns x 4 rows
cols, gap = 5, 6
th, tw = tiles[0].shape[:2]
rows = math.ceil(len(tiles) / cols)
sheet = np.full((rows * th + (rows + 1) * gap, cols * tw + (cols + 1) * gap, 3), 15, np.uint8)
for i, tile in enumerate(tiles):
    r, cc = divmod(i, cols)
    y = gap + r * (th + gap)
    x = gap + cc * (tw + gap)
    sheet[y:y + th, x:x + tw] = tile
sheet_path = os.environ.get("DRONET_EXAMPLES_SHEET", os.path.join(PROJECT_ROOT, "data", "dronet_results", "examples_contact_sheet.png"))
cv2.imwrite(sheet_path, sheet)
print(f"Wrote {len(tiles)} tiles to {OUT}")
print(f"Contact sheet: {sheet_path}  ({sheet.shape[1]}x{sheet.shape[0]})")
