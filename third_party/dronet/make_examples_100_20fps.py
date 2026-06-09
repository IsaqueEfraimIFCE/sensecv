"""Produce 100 continuous DroNet example tiles sampled at 20 FPS.

Each tile shows the 200x200 crop DroNet actually receives plus the model output.
The source clips are treated as one concatenated timeline because no individual
export is long enough for 100 continuous samples at 20 FPS.
"""
import glob
import math
import os

import cv2
import numpy as np
import torch

from dronet_model import load_dronet, preprocess_bgr


DRONET_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(DRONET_DIR))
WEIGHTS = os.environ.get("DRONET_WEIGHTS", os.path.join(DRONET_DIR, "repo", "model", "model_weights.h5"))
BASE = os.environ.get("PILOTGURU_EXPORTS_DIR", os.path.join(PROJECT_ROOT, "data", "exports"))
OUT = os.environ.get("DRONET_EXAMPLES_100_DIR", os.path.join(PROJECT_ROOT, "data", "dronet_results", "examples_100_20fps"))
SHEET_PATH = os.environ.get("DRONET_EXAMPLES_100_SHEET", os.path.join(PROJECT_ROOT, "data", "dronet_results", "examples_100_20fps_contact_sheet.png"))

SAMPLE_FPS = 20.0
N_SAMPLES = 100


def readable_clips():
    required_duration = (N_SAMPLES - 1) / SAMPLE_FPS
    clips = []
    total_duration = 0.0
    for vpath in sorted(glob.glob(os.path.join(BASE, "**", "*.mp4"), recursive=True)):
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            cap.release()
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if fps <= 0 or frames <= 0:
            continue
        duration = frames / fps
        clips.append(
            {
                "path": vpath,
                "fps": fps,
                "frames": frames,
                "duration": duration,
                "start": total_duration,
                "end": total_duration + duration,
            }
        )
        total_duration += duration

    if not clips:
        raise RuntimeError("No readable videos found.")

    if total_duration < required_duration:
        raise RuntimeError(
            "The readable clips are not long enough for "
            f"{N_SAMPLES} continuous samples at {SAMPLE_FPS:g} FPS "
            f"({required_duration:.2f}s required, {total_duration:.2f}s available)."
        )

    return clips, total_duration


def locate_sample(clips, time_s):
    for clip in clips:
        if time_s < clip["end"]:
            local_t = max(0.0, time_s - clip["start"])
            frame_idx = min(clip["frames"] - 1, int(round(local_t * clip["fps"])))
            return clip, frame_idx
    clip = clips[-1]
    return clip, clip["frames"] - 1


def grab_frame(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    return frame if ok else None


def main():
    os.makedirs(OUT, exist_ok=True)
    model = load_dronet(WEIGHTS)

    clips, total_duration = readable_clips()
    sample_times = [i / SAMPLE_FPS for i in range(N_SAMPLES)]
    caps = {}
    tiles = []

    tile_w, img_size = 240, 200
    panel_h = 86

    for n, sample_t in enumerate(sample_times):
        clip, frame_idx = locate_sample(clips, sample_t)
        vpath = clip["path"]
        folder = os.path.basename(os.path.dirname(vpath))
        stem = os.path.splitext(os.path.basename(vpath))[0]
        if vpath not in caps:
            caps[vpath] = cv2.VideoCapture(vpath)

        frame = grab_frame(caps[vpath], frame_idx)
        if frame is None:
            continue

        t, crop = preprocess_bgr(frame)
        with torch.no_grad():
            s, c = model(t)
        s, c = float(s.item()), float(c.item())
        yaw = s * 90.0
        direction = "STRAIGHT" if abs(s) < 0.1 else ("RIGHT" if s > 0 else "LEFT")
        coll_label = "COLLISION" if c >= 0.5 else "CLEAR"
        local_time_s = frame_idx / clip["fps"]

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
            f"#{n+1:03d} clip {folder} f{frame_idx}",
            (6, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"t={sample_t:05.2f}s local={local_time_s:04.2f}s",
            (6, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"steer {s:+.2f} ({yaw:+.0f}d) {direction}",
            (6, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        ccol = (0, 0, 255) if c >= 0.5 else (0, 220, 220)
        cv2.putText(
            panel,
            f"coll {c:.2f} {coll_label}",
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

        tile = np.vstack([img, panel])
        out_path = os.path.join(OUT, f"sample_{n+1:03d}_{folder}_{stem}_f{frame_idx}.png")
        cv2.imwrite(out_path, tile)
        tiles.append(tile)

    for cap in caps.values():
        cap.release()

    if not tiles:
        raise RuntimeError("No tiles were created.")

    cols, gap = 10, 6
    th, tw = tiles[0].shape[:2]
    rows = math.ceil(len(tiles) / cols)
    sheet = np.full(
        (rows * th + (rows + 1) * gap, cols * tw + (cols + 1) * gap, 3),
        15,
        np.uint8,
    )
    for i, tile in enumerate(tiles):
        r, cc = divmod(i, cols)
        y = gap + r * (th + gap)
        x = gap + cc * (tw + gap)
        sheet[y:y + th, x:x + tw] = tile
    cv2.imwrite(SHEET_PATH, sheet)

    print(f"Sources: {len(clips)} readable clips, {total_duration:.2f}s total")
    print(f"Wrote {len(tiles)} tiles to {OUT}")
    print(f"Contact sheet: {SHEET_PATH} ({sheet.shape[1]}x{sheet.shape[0]})")


if __name__ == "__main__":
    main()
