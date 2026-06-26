# -*- coding: utf-8 -*-
"""Build the two-head Kaggle dataset (224x224) from the *revised* deviation cuts.

Labels come from each clip's post-revision cut class (the Revisao tab's effective
label: detected event_type/side + any manual override in review_labels.json,
with excluded clips dropped) -- NOT from the Coleta spreadsheets. See
[[kaggle-two-head-dataset]] and the dataset-labels-from-revision note.

Two heads per image:
  obstacle_class:  0 = no obstacle (caminhada livre),     1 = obstacle (desvio/parada)
  deviation_class: 0 = left (esquerda), 1 = right (direita), 2 = none

Revised-cut class -> (obstacle, deviation):
  desvio LEFT  -> (1, 0)
  desvio RIGHT -> (1, 1)
  parada       -> (1, 2)
  livre        -> (0, 2)

Sources (only datasets dated on/before --until, default 2026-06-20):
  data/derived/manifest_exports/<group>/deviation/<clip>/<clip>.mp4
  data/derived/manifest_exports/<group>/deviation/review_index.json  (built on demand)

Each clip's frames are de-duplicated with grayscale SSIM (threshold 0.97, max gap
0.5 s), centre-cropped to square and saved at 224x224 JPG.

Output:
  data/derived/kaggle_two_head_dataset/
    dataset/*.jpg
    labels.txt                # "file_name obstacle_class deviation_class"
    labels_with_source.tsv
    class_map.json
"""
import argparse
import json
import re
import shutil
import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sensecv import app  # noqa: E402
from sensecv.app import _ssim_gray  # noqa: E402

MANIFEST_ROOT = PROJECT_ROOT / "data" / "derived" / "manifest_exports"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "derived" / "kaggle_two_head_dataset"
DEFAULT_UNTIL = date(2026, 6, 20)
IMG_SIZE = 224
SSIM_THRESHOLD = 0.97
SSIM_MAX_GAP_SEC = 0.5

# Revised cut class -> (obstacle_class, deviation_class), or None to skip.
LABEL_MAP = {
    ("desvio", "LEFT"):  (1, 0),
    ("desvio", "RIGHT"): (1, 1),
    ("parada", "NONE"):  (1, 2),
    ("livre",  "NONE"):  (0, 2),
}
_DATE_RE = re.compile(r"(\d{2})-(\d{2})-(\d{4})")


def group_date(name):
    m = _DATE_RE.search(name)
    if not m:
        return None
    d, mo, y = (int(x) for x in m.groups())
    try:
        return date(y, mo, d)
    except ValueError:
        return None


def label_for(clip):
    et = (clip.get("event_type") or "").strip().lower()
    side = (clip.get("side") or "").strip().upper()
    if et == "desvio":
        return LABEL_MAP.get(("desvio", side))
    return LABEL_MAP.get((et, "NONE"))


def square_224(frame):
    h, w = frame.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    sq = frame[y0:y0 + s, x0:x0 + s]
    return cv2.resize(sq, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def gray_for_ssim(frame):
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90),
                      interpolation=cv2.INTER_AREA)


def extract_clip_frames(video_path, out_dir, base_name, quality):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], {"status": "decode_failed", "frames_before": 0, "images_after": 0}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    max_gap = max(1, int(round(fps * SSIM_MAX_GAP_SEC))) if fps > 0 else 15
    kept, last_gray, last_kept_idx, last_frame, last_idx, idx = [], None, None, None, -1, -1
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        idx += 1
        last_frame, last_idx = frame, idx
        gray = gray_for_ssim(frame)
        keep = (last_gray is None or _ssim_gray(last_gray, gray) < SSIM_THRESHOLD
                or (idx - last_kept_idx) >= max_gap)
        if keep:
            fn = f"{base_name}__frame_{idx:05d}.jpg"
            cv2.imwrite(str(out_dir / fn), square_224(frame),
                        [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            kept.append((fn, idx))
            last_gray, last_kept_idx = gray, idx
    if last_frame is not None and kept and kept[-1][1] != last_idx:
        fn = f"{base_name}__frame_{last_idx:05d}.jpg"
        cv2.imwrite(str(out_dir / fn), square_224(last_frame),
                    [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        kept.append((fn, last_idx))
    cap.release()
    return kept, {"status": "ok" if kept else "no_frames",
                  "frames_before": idx + 1, "images_after": len(kept)}


def collect_clips(until):
    """Every non-excluded revised clip from day<=until deviation groups."""
    clips = []
    if not MANIFEST_ROOT.is_dir():
        return clips
    for group_dir in sorted(p for p in MANIFEST_ROOT.iterdir() if p.is_dir()):
        gd = group_date(group_dir.name)
        if gd is None or gd > until:
            print(f"[pula] {group_dir.name} (data {gd})")
            continue
        dev_dir = group_dir / "deviation"
        if not dev_dir.is_dir():
            continue
        index = app.build_review_index(str(dev_dir))
        if not index:
            print(f"[aviso] sem review_index em {dev_dir}")
            continue
        for clip in index.get("clips", []):
            if clip.get("excluded"):
                continue
            lbl = label_for(clip)
            if lbl is None:
                print(f"[aviso] rotulo desconhecido {group_dir.name}/{clip['folder']}: "
                      f"{clip.get('event_type')}/{clip.get('side')}")
                continue
            video = dev_dir / clip["folder"] / (clip["folder"] + ".mp4")
            if not video.is_file():
                print(f"[aviso] mp4 ausente: {video}")
                continue
            clips.append({"group": group_dir.name, "folder": clip["folder"],
                          "video": video, "obstacle": lbl[0], "deviation": lbl[1],
                          "label": clip.get("label", ""),
                          "overridden": bool(clip.get("label_overridden"))})
    return clips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", default=DEFAULT_UNTIL.isoformat(),
                    help="incluir datasets com data <= esta (YYYY-MM-DD)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--jpeg-quality", type=int, default=95)
    args = ap.parse_args()
    until = date.fromisoformat(args.until)
    out_dir = Path(args.out_dir)
    images_dir = out_dir / "dataset"

    clips = collect_clips(until)
    if not clips:
        raise SystemExit("Nenhum clipe revisado encontrado para o periodo.")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    images_dir.mkdir(parents=True)

    label_rows, clip_summaries = [], []
    for i, clip in enumerate(clips, 1):
        base = f"{clip['group']}__{clip['folder']}"
        kept, summary = extract_clip_frames(clip["video"], images_dir, base, args.jpeg_quality)
        for fn, _ in kept:
            label_rows.append({
                "file_name": f"dataset/{fn}",
                "obstacle_class": clip["obstacle"],
                "deviation_class": clip["deviation"],
                "source_group": clip["group"],
                "source_clip": clip["folder"],
                "source_label": clip["label"],
                "overridden": int(clip["overridden"]),
            })
        summary.update(group=clip["group"], clip=clip["folder"],
                       obstacle=clip["obstacle"], deviation=clip["deviation"])
        clip_summaries.append(summary)
        print(f"[{i:03d}/{len(clips):03d}] {summary['status']} {base}: "
              f"{summary['images_after']} imgs  (o={clip['obstacle']} d={clip['deviation']})")

    with (out_dir / "labels.txt").open("w", encoding="utf-8", newline="\n") as f:
        f.write("file_name obstacle_class deviation_class\n")
        for r in label_rows:
            f.write(f"{r['file_name']} {r['obstacle_class']} {r['deviation_class']}\n")

    headers = ["file_name", "obstacle_class", "deviation_class",
               "source_group", "source_clip", "source_label", "overridden"]
    with (out_dir / "labels_with_source.tsv").open("w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(headers) + "\n")
        for r in label_rows:
            f.write("\t".join(str(r[h]) for h in headers) + "\n")

    o_counts, d_counts, joint = {}, {}, {}
    for r in label_rows:
        o, d = str(r["obstacle_class"]), str(r["deviation_class"])
        o_counts[o] = o_counts.get(o, 0) + 1
        d_counts[d] = d_counts.get(d, 0) + 1
        joint[f"{o}_{d}"] = joint.get(f"{o}_{d}", 0) + 1

    payload = {
        "until": until.isoformat(),
        "image_size": IMG_SIZE,
        "total_images": len(label_rows),
        "image_files": len(list(images_dir.glob("*.jpg"))),
        "obstacle_class_map": {"0": "no_obstacle_livre", "1": "obstacle"},
        "deviation_class_map": {"0": "left", "1": "right", "2": "none"},
        "obstacle_counts": o_counts,
        "deviation_counts": d_counts,
        "joint_counts": joint,
        "label_source": "post-revision cut label (review_index.json)",
        "clips": len(clips),
        "ssim_threshold": SSIM_THRESHOLD,
        "ssim_max_gap_sec": SSIM_MAX_GAP_SEC,
        "clip_summaries": clip_summaries,
    }
    (out_dir / "class_map.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nPronto: {len(label_rows)} imagens de {len(clips)} clipes -> {out_dir}")
    print("obstacle_counts:", json.dumps(o_counts))
    print("deviation_counts:", json.dumps(d_counts))
    print("joint_counts:", json.dumps(joint))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
