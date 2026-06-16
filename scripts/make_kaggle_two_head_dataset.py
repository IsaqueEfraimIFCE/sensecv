# -*- coding: utf-8 -*-
"""Build the Kaggle two-head dataset from the manifest exports plus the clear set.

Sources:
  data/derived/manifest_exports/<dataset>/lateral/<clip>/<clip>.mp4  -> obstacle frames
  data/derived/manifest_exports/<dataset>/lateral/sources.csv        -> labels per clip
  data/derived/clear/images/*.jpg                                    -> clear frames

Output (the folder uploaded to Kaggle, same shape train_two_head_kaggle.py expects):
  data/derived/kaggle_two_head_dataset/
    dataset/*.jpg
    labels.txt               # "file_name obstacle_class deviation_class"
    labels_with_source.tsv
    class_map.json

Classes:
  obstacle_class: 0 = clear, 1 = obstacle
  deviation_class: 0 = left (esquerda), 1 = right (direita), 2 = none

Obstacle frames are de-duplicated per clip with grayscale SSIM (threshold 0.97,
max gap 0.5 s), matching the clear-set build.
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sensecv.app import _ssim_gray  # noqa: E402

DEFAULT_MANIFEST_ROOT = PROJECT_ROOT / "data" / "derived" / "manifest_exports"
DEFAULT_CLEAR_DIR = PROJECT_ROOT / "data" / "derived" / "clear" / "images"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "derived" / "kaggle_two_head_dataset"

SSIM_THRESHOLD = 0.97
SSIM_MAX_GAP_SEC = 0.5


def deviation_class(deviation_side):
    side = (deviation_side or "").strip().lower()
    if "esquerda" in side:
        return 0
    if "direita" in side:
        return 1
    return 2


def gray_for_ssim(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)


def extract_clip_frames(video_path, out_dir, base_name, quality):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], {"status": "decode_failed", "frames_before": 0, "images_after": 0}

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    max_gap_frames = max(1, int(round(fps * SSIM_MAX_GAP_SEC))) if fps > 0 else 15

    kept = []
    last_gray = None
    last_kept_idx = None
    last_frame = None
    last_idx = -1
    idx = -1
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        idx += 1
        last_frame, last_idx = frame, idx
        gray = gray_for_ssim(frame)
        keep = (
            last_gray is None
            or _ssim_gray(last_gray, gray) < SSIM_THRESHOLD
            or (idx - last_kept_idx) >= max_gap_frames
        )
        if keep:
            file_name = f"{base_name}__frame_{idx:05d}.jpg"
            cv2.imwrite(str(out_dir / file_name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            kept.append((file_name, idx))
            last_gray, last_kept_idx = gray, idx
    # Always keep the last frame, like ssim_frame_selection() does.
    if last_frame is not None and kept and kept[-1][1] != last_idx:
        file_name = f"{base_name}__frame_{last_idx:05d}.jpg"
        cv2.imwrite(str(out_dir / file_name), last_frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        kept.append((file_name, last_idx))
    cap.release()
    return kept, {"status": "ok" if kept else "no_frames",
                  "frames_before": idx + 1, "images_after": len(kept)}


def load_obstacle_clips(manifest_root):
    clips = []
    for dataset_dir in sorted(p for p in manifest_root.iterdir() if p.is_dir()):
        sources_csv = dataset_dir / "lateral" / "sources.csv"
        if not sources_csv.is_file():
            continue
        with sources_csv.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status") != "ok" or not row.get("output_folder"):
                    continue
                folder = dataset_dir / "lateral" / row["output_folder"]
                video = folder / f"{row['output_folder']}.mp4"
                if not video.is_file():
                    print(f"[AVISO] mp4 ausente: {video}")
                    continue
                clips.append({
                    "dataset": dataset_dir.name,
                    "folder": row["output_folder"],
                    "video": video,
                    "deviation_class": deviation_class(row.get("deviation_side")),
                    "label": row.get("label") or row["output_folder"],
                })
    return clips


def load_clear_images(clear_dir):
    rows = []
    for jpg in sorted(clear_dir.glob("*.jpg")):
        rows.append({"src": jpg, "file_name": f"clear__{jpg.name}"})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", default=str(DEFAULT_MANIFEST_ROOT))
    parser.add_argument("--clear-dir", default=str(DEFAULT_CLEAR_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()

    manifest_root = Path(args.manifest_root)
    clear_dir = Path(args.clear_dir)
    out_dir = Path(args.out_dir)
    images_dir = out_dir / "dataset"

    obstacle_clips = load_obstacle_clips(manifest_root)
    clear_rows = load_clear_images(clear_dir)
    if not obstacle_clips:
        raise SystemExit(f"Nenhum clipe ok em {manifest_root}")
    if not clear_rows:
        raise SystemExit(f"Nenhuma imagem clear em {clear_dir}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    images_dir.mkdir(parents=True)

    label_rows = []
    clip_summaries = []
    for i, clip in enumerate(obstacle_clips, start=1):
        base_name = f"obstacle__{clip['dataset']}__{clip['folder']}"
        kept, summary = extract_clip_frames(clip["video"], images_dir, base_name, args.jpeg_quality)
        for file_name, _ in kept:
            label_rows.append({
                "file_name": f"dataset/{file_name}",
                "obstacle_class": 1,
                "deviation_class": clip["deviation_class"],
                "source_dataset": clip["dataset"],
                "source_label": clip["label"],
            })
        summary.update({"dataset": clip["dataset"], "clip": clip["folder"],
                        "deviation_class": clip["deviation_class"]})
        clip_summaries.append(summary)
        print(f"[{i:03d}/{len(obstacle_clips):03d}] {summary['status']} "
              f"{clip['dataset']}/{clip['folder']}: {summary['images_after']} imagens")

    for row in clear_rows:
        shutil.copy2(row["src"], images_dir / row["file_name"])
        label_rows.append({
            "file_name": f"dataset/{row['file_name']}",
            "obstacle_class": 0,
            "deviation_class": 2,
            "source_dataset": "clear",
            "source_label": "clear",
        })

    with (out_dir / "labels.txt").open("w", encoding="utf-8", newline="\n") as f:
        f.write("file_name obstacle_class deviation_class\n")
        for row in label_rows:
            f.write(f"{row['file_name']} {row['obstacle_class']} {row['deviation_class']}\n")

    headers = ["file_name", "obstacle_class", "deviation_class", "source_dataset", "source_label"]
    with (out_dir / "labels_with_source.tsv").open("w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(headers) + "\n")
        for row in label_rows:
            f.write("\t".join(str(row[h]) for h in headers) + "\n")

    obstacle_counts = {}
    deviation_counts = {}
    joint_counts = {}
    for row in label_rows:
        o, d = str(row["obstacle_class"]), str(row["deviation_class"])
        obstacle_counts[o] = obstacle_counts.get(o, 0) + 1
        deviation_counts[d] = deviation_counts.get(d, 0) + 1
        joint_counts[f"{o}_{d}"] = joint_counts.get(f"{o}_{d}", 0) + 1

    payload = {
        "total_images": len(label_rows),
        "image_files": len(list(images_dir.glob("*.jpg"))),
        "obstacle_class_map": {"0": "no_obstacle_clear", "1": "obstacle"},
        "deviation_class_map": {"0": "left", "1": "right", "2": "none"},
        "obstacle_counts": obstacle_counts,
        "deviation_counts": deviation_counts,
        "joint_counts": joint_counts,
        "sources": {
            "manifest_exports": str(manifest_root),
            "clear_images": str(clear_dir),
        },
        "obstacle_clips": len(obstacle_clips),
        "ssim_threshold": SSIM_THRESHOLD,
        "ssim_max_gap_sec": SSIM_MAX_GAP_SEC,
        "clips": clip_summaries,
    }
    (out_dir / "class_map.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nDone: {len(label_rows)} imagens "
          f"({sum(1 for r in label_rows if r['obstacle_class'] == 1)} obstacle de "
          f"{len(obstacle_clips)} clipes + "
          f"{sum(1 for r in label_rows if r['obstacle_class'] == 0)} clear) -> {out_dir}")
    print("joint_counts:", json.dumps(joint_counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
