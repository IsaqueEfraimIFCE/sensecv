"""Build a clear/no-obstacle dataset from SSIM-selected final-second frames.

The output follows the existing Kaggle two-head dataset shape:

  data/derived/clear_ssim095_last1s_until_112/
    dataset/*.jpg
    labels.txt
    labels_with_source.tsv
    class_map.json

All generated images are labeled as:
  obstacle_class = 0
  deviation_class = 2
"""
import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sensecv import app  # noqa: E402


DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "derived" / "clear_ssim095_last1s_until_112"


def safe_slug(value):
    value = value.lower().replace("\\", "/")
    value = re.sub(r"[^a-z0-9._/-]+", "_", value)
    value = value.replace("/", "__")
    value = re.sub(r"_+", "_", value).strip("._-")
    return value or "clip"


def selected_uploaded_clips(limit):
    app.refresh_clips()
    rows = []
    uploads_root = Path(app.UPLOADS_DIR).resolve()
    for idx, display in enumerate(app.CLIPS):
        src_dir = Path(app.CLIP_PATHS[display]).resolve()
        try:
            if os.path.commonpath([str(uploads_root), str(src_dir)]) != str(uploads_root):
                continue
        except ValueError:
            continue
        rows.append((idx, display, src_dir, Path(app.CLIP_VIDEO_PATHS[display])))
        if len(rows) >= limit:
            break
    return rows


def video_metadata(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 0 or frame_count <= 0:
        return None
    return {
        "fps": fps,
        "frames": frame_count,
        "duration_s": frame_count / fps,
    }


def decode_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def sample_clip(clip_idx, display, video_path, out_images_dir, threshold, quality):
    meta = video_metadata(video_path)
    if meta is None:
        return [], {"status": "invalid_video", "images": 0}

    start_s = max(0.0, meta["duration_s"] - 1.0)
    end_s = meta["duration_s"]
    selection = app.ssim_frame_selection(
        clip_idx,
        start_s,
        end_s,
        threshold=threshold,
    )
    if selection.get("error"):
        return [], {
            "status": f"ssim_error: {selection['error']}",
            "fps": meta["fps"],
            "frames": meta["frames"],
            "duration_s": meta["duration_s"],
            "window_start_s": start_s,
            "window_end_s": end_s,
            "frames_before_ssim": selection.get("frames_before", 0),
            "images": 0,
        }

    slug = safe_slug(display)
    rows = []
    for selected in selection.get("selected_frames", []):
        time_s = float(selected["time_s"])
        frame_idx = min(meta["frames"] - 1, max(0, int(round(time_s * meta["fps"]))))
        frame = decode_frame(video_path, frame_idx)
        if frame is None:
            continue
        millis = int(round(time_s * 1000.0))
        file_name = f"clear__{slug}__t_{millis:07d}ms__frame_{frame_idx:06d}.jpg"
        out_path = out_images_dir / file_name
        cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        rows.append(
            {
                "file_name": f"dataset/{file_name}",
                "obstacle_class": 0,
                "deviation_class": 2,
                "source_display": display,
                "source_video": str(video_path),
                "time_s": round(time_s, 3),
                "frame_index": frame_idx,
                "source_index": selected.get("source_index", ""),
                "source_frame_id": selected.get("frame_id", ""),
                "ssim_prev": selected.get("ssim_prev", ""),
                "source_fps": meta["fps"],
                "source_frames": meta["frames"],
                "source_duration_s": meta["duration_s"],
                "window_start_s": start_s,
                "window_end_s": end_s,
            }
        )

    return rows, {
        "status": "ok",
        "fps": meta["fps"],
        "frames": meta["frames"],
        "duration_s": meta["duration_s"],
        "window_start_s": start_s,
        "window_end_s": end_s,
        "frames_before_ssim": selection.get("frames_before", 0),
        "frames_after_ssim": selection.get("frames_after", len(rows)),
        "ssim_threshold": selection.get("ssim_threshold", threshold),
        "ssim_max_gap_sec": selection.get("ssim_max_gap_sec", ""),
        "images": len(rows),
    }


def write_outputs(out_dir, image_rows, clip_summaries, selected, threshold):
    labels_path = out_dir / "labels.txt"
    with labels_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("file_name obstacle_class deviation_class\n")
        for row in image_rows:
            f.write(f"{row['file_name']} {row['obstacle_class']} {row['deviation_class']}\n")

    headers = [
        "file_name",
        "obstacle_class",
        "deviation_class",
        "source_display",
        "source_video",
        "time_s",
        "frame_index",
        "source_index",
        "source_frame_id",
        "ssim_prev",
        "source_fps",
        "source_frames",
        "source_duration_s",
        "window_start_s",
        "window_end_s",
    ]
    source_path = out_dir / "labels_with_source.tsv"
    with source_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(headers) + "\n")
        for row in image_rows:
            f.write("\t".join(str(row[h]) for h in headers) + "\n")

    payload = {
        "total_images": len(image_rows),
        "clips_selected": len(selected),
        "clips_ok": sum(1 for item in clip_summaries if item["status"] == "ok"),
        "clips_failed": sum(1 for item in clip_summaries if item["status"] != "ok"),
        "sampling": "SSIM-selected frames from the last 1 second of each video",
        "ssim_threshold": threshold,
        "obstacle_class_map": {"0": "no_obstacle_clear", "1": "obstacle"},
        "deviation_class_map": {"0": "left", "1": "right", "2": "none"},
        "obstacle_counts": {"0": len(image_rows)},
        "deviation_counts": {"2": len(image_rows)},
        "joint_counts": {"0_2": len(image_rows)},
        "clips": clip_summaries,
    }
    (out_dir / "class_map.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=112, help="Number of uploaded videos to sample.")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    images_dir = out_dir / "dataset"

    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output exists; use --overwrite: {out_dir}")
        shutil.rmtree(out_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    selected = selected_uploaded_clips(args.limit)
    if len(selected) < args.limit:
        raise SystemExit(f"Only found {len(selected)} uploaded clips; requested {args.limit}.")

    image_rows = []
    clip_summaries = []
    for ordinal, (clip_idx, display, src_dir, video_path) in enumerate(selected, start=1):
        rows, summary = sample_clip(
            clip_idx,
            display,
            video_path,
            images_dir,
            args.threshold,
            args.jpeg_quality,
        )
        image_rows.extend(rows)
        summary.update(
            {
                "ordinal": ordinal,
                "source_display": display,
                "source_dir": str(src_dir),
                "source_video": str(video_path),
            }
        )
        clip_summaries.append(summary)
        print(
            f"[{ordinal:03d}/{len(selected):03d}] {summary['status']} "
            f"{display}: {summary['images']} images"
        )

    write_outputs(out_dir, image_rows, clip_summaries, selected, args.threshold)
    print(f"Done: {len(image_rows)} images from {len(selected)} clips -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
