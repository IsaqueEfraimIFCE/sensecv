"""Build clear/no-obstacle datasets from metric-selected final-second frames.

Same pipeline as scripts/make_clear_ssim_last1s_dataset.py, but the
frame-to-frame similarity metric is pluggable:

  --metric dinov2   cosine similarity between DINOv2 ViT-S/14 CLS embeddings
  --metric lpips    1 - LPIPS(AlexNet) perceptual distance
  --metric vif      pixel-domain Visual Information Fidelity (sewar vifp)

Selection rule matches app.ssim_frame_selection: walk the frames.json records
inside the last second of each uploaded video, keep a frame when its
similarity to the previously kept frame falls below the threshold (or the
forced max gap elapses), always keep the first and last frames.

Output shape follows the existing Kaggle two-head dataset:

  data/derived/clear_<metric><thr>_last1s_until_<limit>/
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
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sensecv import app  # noqa: E402


DEFAULT_THRESHOLDS = {
    "dinov2": 0.98,
    "lpips": 0.95,
    "vif": 0.95,
}
MAX_GAP_SEC = 0.5


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


# ─── Metric backends ─────────────────────────────────────────────────────────
# Each backend exposes:
#   prepare(bgr_frame) -> representation
#   similarity(prev_repr, repr) -> float in roughly [0, 1], 1 = identical


class Dinov2Metric:
    name = "dinov2"

    def __init__(self):
        import torch

        self.torch = torch
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.model.eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def prepare(self, frame):
        rgb = cv2.cvtColor(cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        tensor = self.torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - self.mean) / self.std
        with self.torch.no_grad():
            emb = self.model(tensor.unsqueeze(0))[0]
        return emb / emb.norm()

    def similarity(self, prev, cur):
        return float((prev * cur).sum())


class LpipsMetric:
    name = "lpips"

    def __init__(self):
        import torch
        import lpips

        self.torch = torch
        self.net = lpips.LPIPS(net="alex", verbose=False)
        self.net.eval()

    def prepare(self, frame):
        rgb = cv2.cvtColor(cv2.resize(frame, (256, 144), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        tensor = self.torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return (tensor * 2.0 - 1.0).unsqueeze(0)

    def similarity(self, prev, cur):
        with self.torch.no_grad():
            dist = float(self.net(prev, cur))
        return 1.0 - dist


class VifMetric:
    name = "vif"

    def __init__(self):
        from sewar.full_ref import vifp

        self.vifp = vifp

    def prepare(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)

    def similarity(self, prev, cur):
        return float(self.vifp(prev, cur))


METRICS = {m.name: m for m in (Dinov2Metric, LpipsMetric, VifMetric)}


def metric_frame_selection(metric, video_path, records, meta, threshold,
                           max_gap_sec=MAX_GAP_SEC):
    """Mirror of app.ssim_frame_selection with a pluggable similarity metric."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, "video unreadable"

    selected = []
    frames_by_no = {}
    prev_repr = None
    prev_time = None
    last_record_i = len(records) - 1
    for record_i, (source_index, frame_rec, time_s) in enumerate(records):
        frame_no = max(0, min(meta["frames"] - 1, int(round(time_s * meta["fps"]))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        if not ok:
            continue

        cur = metric.prepare(frame)
        score = None if prev_repr is None else metric.similarity(prev_repr, cur)
        forced_gap = prev_time is not None and (time_s - prev_time) >= max_gap_sec
        keep = (
            prev_repr is None
            or record_i == last_record_i
            or score is None
            or score < threshold
            or forced_gap
        )
        if keep:
            selected.append({
                "source_index": int(source_index),
                "frame_id": int(frame_rec.get("frame_id", source_index)),
                "time_s": round(float(time_s), 3),
                "frame_index": frame_no,
                "metric_prev": None if score is None else round(float(score), 5),
            })
            frames_by_no[frame_no] = frame
            prev_repr = cur
            prev_time = time_s
    cap.release()
    return {"selected_frames": selected, "frames_before": len(records),
            "frames_by_no": frames_by_no}, None


def sample_clip(metric, clip_idx, display, video_path, out_images_dir, threshold, quality):
    meta = video_metadata(video_path)
    if meta is None:
        return [], {"status": "invalid_video", "images": 0}

    start_s = max(0.0, meta["duration_s"] - 1.0)
    end_s = meta["duration_s"]
    records = app._frame_records_in_window(clip_idx, start_s, end_s)
    base = {
        "fps": meta["fps"],
        "frames": meta["frames"],
        "duration_s": meta["duration_s"],
        "window_start_s": start_s,
        "window_end_s": end_s,
    }
    if not records:
        return [], {**base, "status": "no_frames_in_window", "frames_before_metric": 0, "images": 0}

    selection, error = metric_frame_selection(metric, video_path, records, meta, threshold)
    if error:
        return [], {**base, "status": f"metric_error: {error}",
                    "frames_before_metric": len(records), "images": 0}

    slug = safe_slug(display)
    rows = []
    for selected in selection["selected_frames"]:
        frame = selection["frames_by_no"].get(selected["frame_index"])
        if frame is None:
            continue
        millis = int(round(selected["time_s"] * 1000.0))
        file_name = f"clear__{slug}__t_{millis:07d}ms__frame_{selected['frame_index']:06d}.jpg"
        cv2.imwrite(str(out_images_dir / file_name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        rows.append(
            {
                "file_name": f"dataset/{file_name}",
                "obstacle_class": 0,
                "deviation_class": 2,
                "source_display": display,
                "source_video": str(video_path),
                "time_s": selected["time_s"],
                "frame_index": selected["frame_index"],
                "source_index": selected["source_index"],
                "source_frame_id": selected["frame_id"],
                "metric_prev": "" if selected["metric_prev"] is None else selected["metric_prev"],
                "source_fps": meta["fps"],
                "source_frames": meta["frames"],
                "source_duration_s": meta["duration_s"],
                "window_start_s": start_s,
                "window_end_s": end_s,
            }
        )

    return rows, {
        **base,
        "status": "ok",
        "frames_before_metric": selection["frames_before"],
        "frames_after_metric": len(rows),
        "metric": metric.name,
        "metric_threshold": threshold,
        "metric_max_gap_sec": MAX_GAP_SEC,
        "images": len(rows),
    }


def write_outputs(out_dir, image_rows, clip_summaries, selected, metric_name, threshold):
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
        "metric_prev",
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
        "sampling": f"{metric_name}-selected frames from the last 1 second of each video",
        "metric": metric_name,
        "metric_threshold": threshold,
        "metric_max_gap_sec": MAX_GAP_SEC,
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
    parser.add_argument("--metric", required=True, choices=sorted(METRICS))
    parser.add_argument("--limit", type=int, default=112, help="Number of uploaded videos to sample.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Similarity threshold; keep frame when similarity < threshold.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    threshold = args.threshold if args.threshold is not None else DEFAULT_THRESHOLDS[args.metric]
    thr_slug = f"{threshold:.2f}".replace("0.", "0").replace(".", "")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (
        PROJECT_ROOT / "data" / "derived" / f"clear_{args.metric}{thr_slug}_last1s_until_{args.limit}"
    )
    images_dir = out_dir / "dataset"

    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output exists; use --overwrite: {out_dir}")
        shutil.rmtree(out_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    selected = selected_uploaded_clips(args.limit)
    if len(selected) < args.limit:
        raise SystemExit(f"Only found {len(selected)} uploaded clips; requested {args.limit}.")

    metric = METRICS[args.metric]()
    image_rows = []
    clip_summaries = []
    for ordinal, (clip_idx, display, src_dir, video_path) in enumerate(selected, start=1):
        rows, summary = sample_clip(
            metric,
            clip_idx,
            display,
            video_path,
            images_dir,
            threshold,
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
            f"{display}: {summary['images']} images",
            flush=True,
        )

    write_outputs(out_dir, image_rows, clip_summaries, selected, args.metric, threshold)
    print(f"Done: {len(image_rows)} images from {len(selected)} clips -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
