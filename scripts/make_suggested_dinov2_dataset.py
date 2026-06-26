# -*- coding: utf-8 -*-
"""Build a two-head dataset from every clip's suggested cut, de-duplicated with
DINOv2 embeddings.

For each clip the IMU suggestion (`app.suggest_deviation_cut`) gives a cut
window and an event type. Frames inside that window are decoded and
de-duplicated with the DINOv2 ViT-S/14 cosine metric at the calibrated
threshold (0.98 — consecutive-frame cosine sits ~0.975-0.985, see
[[clear-dataset]]), then labeled under the two-head taxonomy:

  event_type   side    -> obstacle_class  deviation_class   meaning
  desvio       LEFT     -> 1              0                 obstacle, deviated left
  desvio       RIGHT    -> 1              1                 obstacle, deviated right
  desvio       (other)  -> 1              2
  parada       -        -> 1              2                 obstacle, stopped
  livre        -        -> 0              2                 clear / free walk

Output (same shape train_two_head_kaggle.py expects):
  data/derived/suggested_dinov2<thr>_two_head_dataset/
    dataset/*.jpg
    labels.txt               # "file_name obstacle_class deviation_class"
    labels_with_source.tsv
    class_map.json
"""
import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path

import cv2  # noqa: F401  (ensures OpenCV is importable up front)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from sensecv import app  # noqa: E402

# Reuse the DINOv2 backend + selection mirror from the clear-set builder.
_clear = importlib.import_module("make_clear_metric_last1s_dataset")
Dinov2Metric = _clear.Dinov2Metric
metric_frame_selection = _clear.metric_frame_selection
video_metadata = _clear.video_metadata
safe_slug = _clear.safe_slug
MAX_GAP_SEC = _clear.MAX_GAP_SEC

DEFAULT_THRESHOLD = 0.98
DEFAULT_OUT_TEMPLATE = "suggested_dinov2{thr}_two_head_dataset"


def label_for(event_type, side):
    """Map an IMU suggestion to (obstacle_class, deviation_class, tag)."""
    if event_type == "desvio":
        if side == "LEFT":
            return 1, 0, "desvio_esq"
        if side == "RIGHT":
            return 1, 1, "desvio_dir"
        return 1, 2, "desvio"
    if event_type == "parada":
        return 1, 2, "parada"
    if event_type == "livre":
        return 0, 2, "livre"
    return 1, 2, event_type or "desconhecido"


def sample_clip(metric, idx, display, video_path, sug, out_images_dir, threshold, quality):
    obstacle_class, deviation_class, tag = label_for(sug.get("event_type"), sug.get("side"))
    start_s, end_s = float(sug["start"]), float(sug["end"])

    meta = video_metadata(video_path)
    if meta is None:
        return [], {"status": "invalid_video", "images": 0}

    records = app._frame_records_in_window(idx, start_s, end_s)
    base = {
        "event_type": sug.get("event_type"),
        "side": sug.get("side"),
        "obstacle_class": obstacle_class,
        "deviation_class": deviation_class,
        "window_start_s": round(start_s, 3),
        "window_end_s": round(end_s, 3),
        "fps": meta["fps"],
    }
    if not records:
        return [], {**base, "status": "no_frames_in_window", "frames_before": 0, "images": 0}

    selection, error = metric_frame_selection(metric, video_path, records, meta, threshold)
    if error:
        return [], {**base, "status": f"metric_error: {error}",
                    "frames_before": len(records), "images": 0}

    slug = safe_slug(display)
    rows = []
    for selected in selection["selected_frames"]:
        frame = selection["frames_by_no"].get(selected["frame_index"])
        if frame is None:
            continue
        millis = int(round(selected["time_s"] * 1000.0))
        file_name = f"{tag}__{slug}__t_{millis:07d}ms__frame_{selected['frame_index']:06d}.jpg"
        cv2.imwrite(str(out_images_dir / file_name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        rows.append({
            "file_name": f"dataset/{file_name}",
            "obstacle_class": obstacle_class,
            "deviation_class": deviation_class,
            "event_type": sug.get("event_type"),
            "side": sug.get("side") or "",
            "source_display": display,
            "source_video": str(video_path),
            "time_s": selected["time_s"],
            "frame_index": selected["frame_index"],
            "metric_prev": "" if selected["metric_prev"] is None else selected["metric_prev"],
            "window_start_s": round(start_s, 3),
            "window_end_s": round(end_s, 3),
        })

    return rows, {**base, "status": "ok",
                  "frames_before": selection["frames_before"],
                  "images": len(rows)}


def write_outputs(out_dir, image_rows, clip_summaries, threshold):
    with (out_dir / "labels.txt").open("w", encoding="utf-8", newline="\n") as f:
        f.write("file_name obstacle_class deviation_class\n")
        for row in image_rows:
            f.write(f"{row['file_name']} {row['obstacle_class']} {row['deviation_class']}\n")

    headers = ["file_name", "obstacle_class", "deviation_class", "event_type", "side",
               "source_display", "source_video", "time_s", "frame_index",
               "metric_prev", "window_start_s", "window_end_s"]
    with (out_dir / "labels_with_source.tsv").open("w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(headers) + "\n")
        for row in image_rows:
            f.write("\t".join(str(row[h]) for h in headers) + "\n")

    obstacle_counts, deviation_counts, joint_counts, event_counts = {}, {}, {}, {}
    for row in image_rows:
        o, d = str(row["obstacle_class"]), str(row["deviation_class"])
        obstacle_counts[o] = obstacle_counts.get(o, 0) + 1
        deviation_counts[d] = deviation_counts.get(d, 0) + 1
        joint_counts[f"{o}_{d}"] = joint_counts.get(f"{o}_{d}", 0) + 1
        et = row["event_type"] or "none"
        event_counts[et] = event_counts.get(et, 0) + 1

    payload = {
        "total_images": len(image_rows),
        "clips_ok": sum(1 for s in clip_summaries if s["status"] == "ok"),
        "clips_skipped": sum(1 for s in clip_summaries if s["status"] != "ok"),
        "sampling": "DINOv2-deduplicated frames from each clip's suggested IMU cut",
        "metric": "dinov2",
        "metric_threshold": threshold,
        "metric_max_gap_sec": MAX_GAP_SEC,
        "obstacle_class_map": {"0": "no_obstacle_clear", "1": "obstacle"},
        "deviation_class_map": {"0": "left", "1": "right", "2": "none"},
        "obstacle_counts": obstacle_counts,
        "deviation_counts": deviation_counts,
        "joint_counts": joint_counts,
        "event_type_counts": event_counts,
        "clips": clip_summaries,
    }
    (out_dir / "class_map.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="DINOv2 cosine threshold; keep a frame when similarity < threshold.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N clips (debug).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    threshold = args.threshold
    thr_slug = f"{threshold:.2f}".replace("0.", "0").replace(".", "")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (
        PROJECT_ROOT / "data" / "derived" / DEFAULT_OUT_TEMPLATE.format(thr=thr_slug)
    )
    images_dir = out_dir / "dataset"
    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output exists; use --overwrite: {out_dir}")
        shutil.rmtree(out_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    app.refresh_clips()
    n = len(app.CLIPS)
    if args.limit:
        n = min(n, args.limit)

    metric = Dinov2Metric()
    image_rows = []
    clip_summaries = []
    for idx in range(n):
        display = app.CLIPS[idx]
        try:
            sug = app.suggest_deviation_cut(idx)
        except Exception as e:
            sug = {"found": False, "message": f"erro: {e}"}
        if not sug.get("found"):
            clip_summaries.append({"ordinal": idx + 1, "source_display": display,
                                   "status": "no_suggestion",
                                   "message": sug.get("message", ""), "images": 0})
            print(f"[{idx + 1:03d}/{n:03d}] skip {display}: {sug.get('message', '')}", flush=True)
            continue

        rows, summary = sample_clip(metric, idx, display, app.clip_video_path(idx),
                                    sug, images_dir, threshold, args.jpeg_quality)
        image_rows.extend(rows)
        summary.update({"ordinal": idx + 1, "source_display": display})
        clip_summaries.append(summary)
        print(f"[{idx + 1:03d}/{n:03d}] {summary['status']} {display} "
              f"[{summary.get('event_type')}/{summary.get('side')}]: {summary['images']} images",
              flush=True)

    write_outputs(out_dir, image_rows, clip_summaries, threshold)

    oc = {}
    for r in image_rows:
        k = f"{r['obstacle_class']}_{r['deviation_class']}"
        oc[k] = oc.get(k, 0) + 1
    print(f"\nDone: {len(image_rows)} images from "
          f"{sum(1 for s in clip_summaries if s['status'] == 'ok')} clips -> {out_dir}")
    print("joint_counts:", json.dumps(oc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
