"""Create DroNet example tiles for SenseCV clips 20 and 21 sampled at 10 FPS."""
import math
import os

import cv2
import numpy as np
import torch

from dronet_model import load_dronet, preprocess_bgr


DRONET_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(DRONET_DIR))
WEIGHTS = os.environ.get("DRONET_WEIGHTS", os.path.join(DRONET_DIR, "repo", "model", "model_weights.h5"))
BASE = os.environ.get(
    "SENSECV_20260527_GIMBAL_ROOT",
    os.path.join(PROJECT_ROOT, "data", "datasets", "SenseCV-27-05-2026-IFCE-Gimbal"),
)
OUT = os.environ.get("DRONET_SENSECV_20_21_DIR", os.path.join(PROJECT_ROOT, "data", "dronet_results", "sensecv_20_21_10fps"))
SHEET_PATH = os.environ.get("DRONET_SENSECV_20_21_SHEET", os.path.join(PROJECT_ROOT, "data", "dronet_results", "sensecv_20_21_10fps_contact_sheet.png"))
SAMPLE_FPS = 10.0
CLIPS = ("20", "21")


def draw_tile(crop, clip, frame_idx, sample_idx, time_s, steering, collision):
    tile_w, img_size, panel_h = 240, 200, 86
    yaw = steering * 90.0
    direction = "STRAIGHT" if abs(steering) < 0.1 else ("RIGHT" if steering > 0 else "LEFT")
    coll_label = "COLLISION" if collision >= 0.5 else "CLEAR"

    img = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    pad = (tile_w - img_size) // 2
    img = cv2.copyMakeBorder(
        img,
        0,
        0,
        pad,
        tile_w - img_size - pad,
        cv2.BORDER_CONSTANT,
        value=(40, 40, 40),
    )

    panel = np.full((panel_h, tile_w, 3), 25, np.uint8)
    cv2.putText(
        panel,
        f"#{sample_idx:03d} clip {clip} f{frame_idx}",
        (6, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"t={time_s:05.2f}s @10fps",
        (6, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"steer {steering:+.2f} ({yaw:+.0f}d) {direction}",
        (6, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    ccol = (0, 0, 255) if collision >= 0.5 else (0, 220, 220)
    cv2.putText(
        panel,
        f"coll {collision:.2f} {coll_label}",
        (6, 74),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        ccol,
        1,
        cv2.LINE_AA,
    )

    cx, cy = tile_w - 28, 65
    ex = int(cx + math.sin(math.radians(yaw)) * 17)
    ey = int(cy - math.cos(math.radians(yaw)) * 17)
    cv2.line(panel, (cx, cy), (ex, ey), (0, 255, 0), 2, cv2.LINE_AA)
    return np.vstack([img, panel])


def sample_clip(model, clip, sample_start):
    vpath = os.path.join(BASE, clip, "video.mp4")
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {vpath}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if src_fps <= 0 or src_frames <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video metadata for {vpath}: fps={src_fps}, frames={src_frames}")

    duration = src_frames / src_fps
    n_samples = int(math.floor(duration * SAMPLE_FPS))
    tiles = []

    for i in range(n_samples):
        time_s = i / SAMPLE_FPS
        frame_idx = min(src_frames - 1, int(round(time_s * src_fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue

        t, crop = preprocess_bgr(frame)
        with torch.no_grad():
            s, c = model(t)
        steering, collision = float(s.item()), float(c.item())
        sample_idx = sample_start + len(tiles)
        tile = draw_tile(crop, clip, frame_idx, sample_idx, time_s, steering, collision)
        out_path = os.path.join(OUT, f"sample_{sample_idx:03d}_clip{clip}_f{frame_idx}.png")
        cv2.imwrite(out_path, tile)
        tiles.append(tile)

    cap.release()
    return tiles, {"clip": clip, "path": vpath, "fps": src_fps, "frames": src_frames, "duration": duration}


def write_contact_sheet(tiles):
    cols, gap = 10, 6
    th, tw = tiles[0].shape[:2]
    rows = math.ceil(len(tiles) / cols)
    sheet = np.full(
        (rows * th + (rows + 1) * gap, cols * tw + (cols + 1) * gap, 3),
        15,
        np.uint8,
    )
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        y = gap + r * (th + gap)
        x = gap + c * (tw + gap)
        sheet[y:y + th, x:x + tw] = tile
    cv2.imwrite(SHEET_PATH, sheet)
    return sheet.shape


def main():
    os.makedirs(OUT, exist_ok=True)
    model = load_dronet(WEIGHTS)

    all_tiles = []
    metas = []
    next_idx = 1
    for clip in CLIPS:
        tiles, meta = sample_clip(model, clip, next_idx)
        all_tiles.extend(tiles)
        metas.append(meta)
        next_idx += len(tiles)

    if not all_tiles:
        raise RuntimeError("No samples were created.")

    sheet_shape = write_contact_sheet(all_tiles)
    for meta in metas:
        print(
            f"Clip {meta['clip']}: {meta['frames']} frames, "
            f"{meta['fps']:.3f} FPS, {meta['duration']:.2f}s"
        )
    print(f"Wrote {len(all_tiles)} tiles to {OUT}")
    print(f"Contact sheet: {SHEET_PATH} ({sheet_shape[1]}x{sheet_shape[0]})")


if __name__ == "__main__":
    main()
