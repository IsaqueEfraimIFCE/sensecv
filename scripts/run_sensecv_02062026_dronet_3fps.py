"""Run DroNet over the 02-06-2026 SenseCV gimbal clips at 3 FPS.

Outputs annotated PNG classifications and contact sheets into this workspace.
"""
import csv
import json
import math
import os
import sys

import cv2
import numpy as np
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRONET_DIR = os.environ.get("DRONET_DIR", os.path.join(PROJECT_ROOT, "third_party", "dronet"))
WEIGHTS = os.environ.get("DRONET_WEIGHTS", os.path.join(DRONET_DIR, "repo", "model", "model_weights.h5"))
DATASET_ROOT = os.environ.get(
    "SENSECV_02062026_GIMBAL_ROOT",
    os.path.join(
        PROJECT_ROOT,
        "data",
        "datasets",
        "SenseCV-02-06-2026-IFCE-Gimbal",
        "SenseCV-02-06-2026-IFCE-Gimbal",
    ),
)
OUT_ROOT = os.environ.get("DRONET_SENSECV_02062026_OUT_DIR", os.path.join(PROJECT_ROOT, "data", "dronet_results", "sensecv_02062026_3fps"))
SAMPLE_FPS = 3.0


sys.path.insert(0, DRONET_DIR)
from dronet_model import load_dronet, preprocess_bgr  # noqa: E402


def safe_name(value):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))


def clip_dirs():
    dirs = []
    for name in sorted(os.listdir(DATASET_ROOT)):
        path = os.path.join(DATASET_ROOT, name)
        video = os.path.join(path, "video.mp4")
        if os.path.isdir(path) and os.path.isfile(video):
            dirs.append((name, video))
    return dirs


def classify(steering, collision):
    yaw = steering * 90.0
    direction = "STRAIGHT" if abs(steering) < 0.1 else ("RIGHT" if steering > 0 else "LEFT")
    collision_label = "COLLISION" if collision >= 0.5 else "CLEAR"
    return yaw, direction, collision_label


def draw_tile(crop, clip, sample_idx, frame_idx, time_s, steering, collision):
    tile_w, img_size, panel_h = 260, 200, 96
    yaw, direction, collision_label = classify(steering, collision)

    img = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    pad = (tile_w - img_size) // 2
    img = cv2.copyMakeBorder(
        img,
        0,
        0,
        pad,
        tile_w - img_size - pad,
        cv2.BORDER_CONSTANT,
        value=(38, 38, 38),
    )

    panel = np.full((panel_h, tile_w, 3), 24, np.uint8)
    lines = [
        (f"clip {clip}  sample {sample_idx:03d}", (235, 235, 235)),
        (f"frame {frame_idx}  t={time_s:06.2f}s @3fps", (210, 210, 210)),
        (f"steer={steering:+.3f} yaw={yaw:+.1f} {direction}", (0, 255, 0)),
        (f"collision={collision:.3f} {collision_label}", (0, 0, 255) if collision >= 0.5 else (0, 220, 220)),
    ]
    for i, (text, color) in enumerate(lines):
        cv2.putText(
            panel,
            text,
            (7, 17 + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            color,
            1,
            cv2.LINE_AA,
        )

    cx, cy = tile_w - 30, 74
    ex = int(cx + math.sin(math.radians(yaw)) * 18)
    ey = int(cy - math.cos(math.radians(yaw)) * 18)
    cv2.line(panel, (cx, cy), (ex, ey), (0, 255, 0), 2, cv2.LINE_AA)
    return np.vstack([img, panel])


def sample_times(duration_s):
    count = int(math.floor(duration_s * SAMPLE_FPS)) + 1
    for i in range(count):
        t = i / SAMPLE_FPS
        if t <= duration_s:
            yield i + 1, t


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_contact_sheet(path, images):
    if not images:
        return None
    cols, gap = 8, 6
    th, tw = images[0].shape[:2]
    rows = math.ceil(len(images) / cols)
    sheet = np.full(
        (rows * th + (rows + 1) * gap, cols * tw + (cols + 1) * gap, 3),
        15,
        np.uint8,
    )
    for i, image in enumerate(images):
        row, col = divmod(i, cols)
        y = gap + row * (th + gap)
        x = gap + col * (tw + gap)
        sheet[y:y + th, x:x + tw] = image
    cv2.imwrite(path, sheet)
    return {"width": int(sheet.shape[1]), "height": int(sheet.shape[0])}


def run_clip(model, clip, video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], {"clip": clip, "status": "unreadable", "samples": 0}

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0 or frame_count <= 0:
        cap.release()
        return [], {"clip": clip, "status": "invalid_metadata", "samples": 0, "fps": fps, "frames": frame_count}

    duration_s = frame_count / fps
    out_dir = os.path.join(OUT_ROOT, safe_name(clip))
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    tiles = []
    for sample_idx, time_s in sample_times(duration_s):
        frame_idx = min(frame_count - 1, int(round(time_s * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue

        tensor, crop = preprocess_bgr(frame)
        with torch.no_grad():
            steering_t, collision_t = model(tensor)

        steering = float(steering_t.item())
        collision = float(collision_t.item())
        yaw, direction, collision_label = classify(steering, collision)
        tile = draw_tile(crop, clip, len(rows) + 1, frame_idx, time_s, steering, collision)

        image_name = f"classification_{len(rows) + 1:03d}_frame_{frame_idx:06d}.png"
        image_path = os.path.join(out_dir, image_name)
        cv2.imwrite(image_path, tile)
        tiles.append(tile)

        rows.append(
            {
                "clip_id": clip,
                "sample": len(rows) + 1,
                "frame": frame_idx,
                "time_s": round(time_s, 4),
                "source_fps": round(fps, 4),
                "sample_fps": SAMPLE_FPS,
                "video_frames": frame_count,
                "resolution": f"{width}x{height}",
                "steering": steering,
                "yaw_deg": yaw,
                "direction": direction,
                "collision_prob": collision,
                "collision_label": collision_label,
                "image": image_path,
            }
        )

    cap.release()
    write_csv(os.path.join(out_dir, "classifications.csv"), rows)
    sheet_meta = write_contact_sheet(os.path.join(out_dir, "contact_sheet.png"), tiles)

    if rows:
        steers = np.array([row["steering"] for row in rows], dtype=np.float32)
        collisions = np.array([row["collision_prob"] for row in rows], dtype=np.float32)
        summary = {
            "clip": clip,
            "status": "ok",
            "samples": len(rows),
            "frames": frame_count,
            "fps": round(fps, 4),
            "duration_s": round(duration_s, 4),
            "resolution": f"{width}x{height}",
            "steering_mean": round(float(steers.mean()), 6),
            "steering_min": round(float(steers.min()), 6),
            "steering_max": round(float(steers.max()), 6),
            "collision_mean": round(float(collisions.mean()), 6),
            "collision_max": round(float(collisions.max()), 6),
            "frac_collision_ge_0.5": round(float((collisions >= 0.5).mean()), 6),
            "contact_sheet": os.path.join(out_dir, "contact_sheet.png"),
            "contact_sheet_size": sheet_meta,
        }
    else:
        summary = {
            "clip": clip,
            "status": "decode_failed",
            "samples": 0,
            "frames": frame_count,
            "fps": round(fps, 4),
            "duration_s": round(duration_s, 4),
            "resolution": f"{width}x{height}",
        }
    return rows, summary


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    model = load_dronet(WEIGHTS)

    all_rows = []
    summaries = []
    clips = clip_dirs()
    for clip, video_path in clips:
        rows, summary = run_clip(model, clip, video_path)
        all_rows.extend(rows)
        summaries.append(summary)
        print(
            f"[{summary['status']}] {clip}: {summary.get('samples', 0)} classifications, "
            f"collision_mean={summary.get('collision_mean', '')}"
        )

    write_csv(os.path.join(OUT_ROOT, "all_classifications.csv"), all_rows)
    with open(os.path.join(OUT_ROOT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_root": DATASET_ROOT,
                "dronet_dir": DRONET_DIR,
                "weights": WEIGHTS,
                "sample_fps": SAMPLE_FPS,
                "clip_folders": len(clips),
                "classifications": len(all_rows),
                "summaries": summaries,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Wrote {len(all_rows)} classifications into {OUT_ROOT}")


if __name__ == "__main__":
    main()
