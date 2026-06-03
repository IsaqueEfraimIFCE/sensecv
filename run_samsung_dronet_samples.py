"""Sample Samsung SenseCV clips and run DroNet inference.

The script uses the local DroNet checkout for the model implementation and
weights, then writes per-ID annotated PNG samples plus CSV/JSON summaries into
this PilotGuru workspace.
"""
import csv
import json
import math
import os
import random
import sys

import cv2
import numpy as np
import torch


DRONET_DIR = r"C:\Users\Isaque\Desktop\dronet"
WEIGHTS = os.path.join(DRONET_DIR, "repo", "model", "model_weights.h5")
SAMSUNG_ROOT = (
    r"C:\Users\Isaque\Downloads\SenseCV-30-05-2026-IFCE-Gimbal-Samsung"
    r"\SenseCV-30-05-2026-IFCE-Gimbal-Samsung"
)
METADATA_CSV = (
    r"C:\Users\Isaque\Downloads"
    r"\Coleta  IFCE - 30 de maio de 2026 - ANDRÉ (Samsung) - Página1.csv"
)
OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dronet_samsung_random_samples")
SAMPLES_PER_CLIP = 20
RANDOM_SEED = 20260530


sys.path.insert(0, DRONET_DIR)
from dronet_model import load_dronet, preprocess_bgr  # noqa: E402


def safe_name(value):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))


def read_metadata():
    rows = {}
    with open(METADATA_CSV, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            id_value = str(row.get("ID", "")).strip()
            if id_value:
                rows[id_value] = row
    return rows


def clip_dirs():
    dirs = []
    for name in sorted(os.listdir(SAMSUNG_ROOT)):
        path = os.path.join(SAMSUNG_ROOT, name)
        video = os.path.join(path, "video.mp4")
        if os.path.isdir(path) and os.path.isfile(video):
            dirs.append((name, path, video))
    return dirs


def draw_tile(frame, crop, folder, frame_idx, time_s, steering, collision, meta):
    src_h, src_w = frame.shape[:2]
    tile_w, img_size, panel_h = 320, 240, 116
    yaw = steering * 90.0
    direction = "STRAIGHT" if abs(steering) < 0.1 else ("RIGHT" if steering > 0 else "LEFT")
    coll_label = "COLLISION" if collision >= 0.5 else "CLEAR"

    img = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
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

    panel = np.full((panel_h, tile_w, 3), 24, np.uint8)
    lines = [
        f"ID {folder}  frame {frame_idx}  t={time_s:.2f}s",
        f"{src_w}x{src_h}  steer={steering:+.3f} yaw={yaw:+.1f} {direction}",
        f"collision={collision:.3f} {coll_label}",
    ]
    local = (meta or {}).get("LOCAL", "").strip()
    if local:
        lines.append(local[:42])

    colors = [(235, 235, 235), (0, 255, 0), (0, 0, 255) if collision >= 0.5 else (0, 220, 220), (210, 210, 210)]
    for i, text in enumerate(lines):
        cv2.putText(
            panel,
            text,
            (8, 20 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            colors[i],
            1,
            cv2.LINE_AA,
        )

    cx, cy = tile_w - 32, 84
    ex = int(cx + math.sin(math.radians(yaw)) * 22)
    ey = int(cy - math.cos(math.radians(yaw)) * 22)
    cv2.line(panel, (cx, cy), (ex, ey), (0, 255, 0), 2, cv2.LINE_AA)
    return np.vstack([img, panel])


def sample_indices(frame_count, folder):
    n = min(SAMPLES_PER_CLIP, frame_count)
    rng = random.Random(f"{RANDOM_SEED}:{folder}")
    return sorted(rng.sample(range(frame_count), n))


def run_clip(model, folder, video_path, metadata):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], {"clip": folder, "status": "unreadable", "frames": 0}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if frame_count <= 0:
        cap.release()
        return [], {"clip": folder, "status": "no_frames", "frames": 0}

    out_dir = os.path.join(OUT_ROOT, safe_name(folder))
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for sample_num, frame_idx in enumerate(sample_indices(frame_count, folder), 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue

        tensor, crop = preprocess_bgr(frame)
        with torch.no_grad():
            steering_t, collision_t = model(tensor)
        steering = float(steering_t.item())
        collision = float(collision_t.item())
        yaw = steering * 90.0
        time_s = frame_idx / fps

        tile = draw_tile(frame, crop, folder, frame_idx, time_s, steering, collision, metadata.get(folder))
        image_name = f"sample_{sample_num:02d}_frame_{frame_idx:06d}.png"
        image_path = os.path.join(out_dir, image_name)
        cv2.imwrite(image_path, tile)

        meta = metadata.get(folder, {})
        rows.append(
            {
                "clip_id": folder,
                "sample": sample_num,
                "frame": frame_idx,
                "time_s": round(time_s, 4),
                "fps": round(fps, 4),
                "video_frames": frame_count,
                "resolution": f"{width}x{height}",
                "steering": steering,
                "yaw_deg": yaw,
                "collision_prob": collision,
                "direction": "STRAIGHT" if abs(steering) < 0.1 else ("RIGHT" if steering > 0 else "LEFT"),
                "collision_label": "COLLISION" if collision >= 0.5 else "CLEAR",
                "local": meta.get("LOCAL", ""),
                "descricao": meta.get("DESCRIÇÃO", meta.get("DESCRICAO", "")),
                "image": image_path,
            }
        )

    cap.release()

    if rows:
        steers = np.array([row["steering"] for row in rows], dtype=np.float32)
        collisions = np.array([row["collision_prob"] for row in rows], dtype=np.float32)
        summary = {
            "clip": folder,
            "status": "ok",
            "samples": len(rows),
            "frames": frame_count,
            "fps": round(fps, 4),
            "resolution": f"{width}x{height}",
            "steering_mean": round(float(steers.mean()), 6),
            "steering_min": round(float(steers.min()), 6),
            "steering_max": round(float(steers.max()), 6),
            "collision_mean": round(float(collisions.mean()), 6),
            "collision_max": round(float(collisions.max()), 6),
            "frac_collision_ge_0.5": round(float((collisions >= 0.5).mean()), 6),
            "has_metadata": folder in metadata,
        }
    else:
        summary = {"clip": folder, "status": "decode_failed", "samples": 0, "frames": frame_count}
    return rows, summary


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    metadata = read_metadata()
    model = load_dronet(WEIGHTS)

    all_rows = []
    summaries = []
    clips = clip_dirs()
    for folder, _path, video_path in clips:
        rows, summary = run_clip(model, folder, video_path, metadata)
        if rows:
            write_csv(os.path.join(OUT_ROOT, safe_name(folder), "samples.csv"), rows)
        all_rows.extend(rows)
        summaries.append(summary)
        print(
            f"[{summary.get('status')}] {folder}: "
            f"{summary.get('samples', 0)} samples, "
            f"collision_mean={summary.get('collision_mean', '')}"
        )

    write_csv(os.path.join(OUT_ROOT, "all_samples.csv"), all_rows)
    with open(os.path.join(OUT_ROOT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "samsung_root": SAMSUNG_ROOT,
                "metadata_csv": METADATA_CSV,
                "clip_folders": len(clips),
                "metadata_rows": len(metadata),
                "samples_per_clip_requested": SAMPLES_PER_CLIP,
                "random_seed": RANDOM_SEED,
                "summaries": summaries,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Wrote {len(all_rows)} sample predictions into {OUT_ROOT}")


if __name__ == "__main__":
    main()
