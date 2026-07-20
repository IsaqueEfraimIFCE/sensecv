# -*- coding: utf-8 -*-
"""Build combined all-frames image datasets for Kaggle from THREE sources.

Like make_kaggle_image_datasets.py this is self-contained (OpenCV only, no
sensecv.app / IMU / history.json) so it runs on Kaggle from the uploaded
folders. It merges every decodable frame from:

  1. --cuts       the labelled `deviation` cut folders (cut_info.json with
                  event_type/side + trimmed <clip>.mp4), e.g. datasetcortado-andre
  2. --endpoints  the `dataset_endpoints/deviation` head/tail cut folders
                  (first/last 0.5 s of every clip; cut_info.json has segment
                  but no event_type)  -> labelled free_path / none
  3. --clear      the clear clip folders (e.g. data/test-clear, one
                  `NN_sem_obstaculos` folder per clip with a top-level .mp4)
                  -> labelled free_path / none

and writes TWO ImageFolder-style datasets (class sub-folders + labels.csv +
dataset_info.json), every frame in both:

  obstacle_freepath/   obstacle vs free_path
  deviation_lrn/       left / right / none

Label mapping for the labelled cuts (project taxonomy):

  event_type   side    obstacle label   deviation label
  desvio       LEFT     obstacle         left
  desvio       RIGHT    obstacle         right
  parada       NONE     obstacle         none
  livre        NONE     free_path        none

Endpoint and clear frames carry `obstacle=free_path`, `deviation=none`.
`labels.csv` gains `source` (cuts/endpoints/clear) and `segment` (head/tail)
columns so any source can be filtered out later. `source_clip` strips the
trailing `_head`/`_tail` so a physical clip's endpoint and cut frames group
together in a clip-based train/val/test split (no leakage).

Kaggle usage (notebook cell; edit CONFIG or pass flags):

    !python make_kaggle_allframes_dataset.py \
        --cuts      /kaggle/input/<ds>/datasetcortado/deviation \
        --endpoints /kaggle/input/<ds>/dataset_endpoints/deviation \
        --clear     /kaggle/input/<ds>/test-clear \
        --out       /kaggle/working/datasets

Local usage:

    python scripts/make_kaggle_allframes_dataset.py \
        --cuts      data/datasetcortado-andre/datasetcortado/deviation \
        --endpoints data/derived/manifest_exports/dataset_endpoints/deviation \
        --clear     data/test-clear \
        --out       data/derived/kaggle_allframes_datasets
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import cv2

# ---------------------------------------------------------------------------
# Notebook config (Kaggle / Colab): pasting this file into a cell can't pass
# CLI flags, so edit these defaults and just run the cell. Command-line flags
# override anything set here.
# ---------------------------------------------------------------------------
CONFIG = {
    "cuts": "/kaggle/input/datasets/isaqueefraim/datasetandre/deviation",
    "endpoints": "/kaggle/input/datasets/isaqueefraim/dataset-endpoints/deviation",
    "clear": "/kaggle/input/datasets/isaqueefraim/test-clear",
    "out": "/kaggle/working/datasets",
    "clear_date": "22-06-2026",  # collection date of the clear clips (DD-MM-YYYY)
    "img_size": 224,             # output side in px (0 = keep native resolution)
    "stride": 1,
    "jpeg_quality": 95,
    "limit": None,
    "overwrite": True,
}

OBSTACLE_CLASSES = ["free_path", "obstacle"]
DEVIATION_CLASSES = ["none", "left", "right"]


def labels_for(event_type, side):
    """(event_type, side) -> (obstacle_label, deviation_label) or (None, None)."""
    et = (event_type or "").strip().lower()
    sd = (side or "").strip().upper()
    if et == "desvio":
        if sd == "LEFT":
            return "obstacle", "left"
        if sd == "RIGHT":
            return "obstacle", "right"
        return "obstacle", "none"
    if et == "parada":
        return "obstacle", "none"
    if et == "livre":
        return "free_path", "none"
    return None, None


def slugify(name):
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in name)


def strip_segment(name):
    """Fold a `<clip>_head` / `<clip>_tail` endpoint folder back to its clip."""
    for suffix in ("_head", "_tail"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


# ---------------------------------------------------------------------------
# Input discovery -> a flat list of jobs {folder, video, labels, meta}
# ---------------------------------------------------------------------------

def load_sources_csv(root):
    """Ledger fallback: output_folder -> row from a root-level sources.csv.

    Both cut exporters write one (make_datasetcortado / make_dataset_endpoints);
    136 datasetcortado-andre folders (incl. 23 of the 33 day-22 clips) carry no
    per-folder cut_info.json but ARE labelled there.
    """
    ledger = {}
    for candidate in (root / "sources.csv", root.parent / "sources.csv"):
        if candidate.is_file():
            with candidate.open(encoding="utf-8", newline="") as f:
                for r in csv.DictReader(f):
                    name = (r.get("output_folder") or "").strip()
                    if name:
                        ledger[name] = r
            break
    return ledger


def find_cut_jobs(root, source, allow_missing_info=False):
    """Cut folders (cut_info.json + .mp4) under `root`, labelled per cut_info.

    A folder without cut_info.json falls back to its sources.csv ledger row.
    With `allow_missing_info` (the endpoints source), a folder missing both is
    still used: the segment is inferred from the `_head`/`_tail` folder suffix
    and the labels default to free path.
    """
    jobs, skipped = [], []
    ledger = load_sources_csv(root)
    # Only sub-folders count as cuts: review/index files sit at the root.
    folders = sorted({m.parent for m in root.rglob("*.mp4")} - {root})
    for folder in folders:
        mp4s = [m for m in sorted(folder.glob("*.mp4"))
                if not m.name.startswith("review")]
        if not mp4s:
            skipped.append({"clip": folder.name, "source": source, "reason": "no_mp4"})
            continue
        ci = folder / "cut_info.json"
        info = None
        if ci.is_file():
            try:
                info = json.loads(ci.read_text(encoding="utf-8"))
            except Exception as e:  # noqa: BLE001
                skipped.append({"clip": folder.name, "source": source,
                                "reason": f"bad cut_info.json: {e}"})
                continue
        elif folder.name in ledger:
            info = ledger[folder.name]
            print(f"[info] {source}: {folder.name} labelled from sources.csv "
                  f"({info.get('event_type') or 'no event'}/"
                  f"{info.get('side') or info.get('segment') or '-'})")
        elif not allow_missing_info:
            skipped.append({"clip": folder.name, "source": source,
                            "reason": "no cut_info.json and no sources.csv row"})
            continue
        if info is None:
            seg = next((s for s in ("head", "tail")
                        if folder.name.endswith("_" + s)), "")
            info = {"segment": seg}
            print(f"[warn] {source}: {folder.name} has no cut_info.json; "
                  f"inferred segment={seg or '?'} (free path)")
        et, sd = info.get("event_type"), info.get("side")
        segment = info.get("segment", "")
        if et:
            obs, dev = labels_for(et, sd)
            if obs is None:
                skipped.append({"clip": folder.name, "source": source,
                                "reason": f"unknown event_type={et}"})
                continue
        else:
            # Endpoint (head/tail) cuts carry no event label: free path.
            obs, dev = "free_path", "none"
        jobs.append({
            "folder": folder, "video": mp4s[0], "source": source,
            "obs": obs, "dev": dev, "event_type": et or "", "side": sd or "",
            "segment": segment, "source_clip": strip_segment(folder.name),
        })
    return jobs, skipped


def find_clear_jobs(root, source="clear", clear_date=""):
    """One job per direct sub-folder of `root` holding a top-level .mp4.

    The clear clips are the day-22 free-path recordings but their folder names
    (`NN_sem_obstaculos`) carry no date, so `clear_date` (DD-MM-YYYY) is
    prefixed onto `source_clip` — the trainer's day-based split then puts them
    in the day-22 test set alongside the labelled day-22 cuts.
    """
    jobs, skipped = [], []
    prefix = f"SenseCV-{clear_date}-clear_" if clear_date else ""
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        if folder.name == "__MACOSX":
            continue
        mp4s = sorted(folder.glob("*.mp4"))  # top level only: skips ssim_review/
        if not mp4s:
            skipped.append({"clip": folder.name, "source": source, "reason": "no_mp4"})
            continue
        named = [m for m in mp4s if m.stem == folder.name]
        jobs.append({
            "folder": folder, "video": (named or mp4s)[0], "source": source,
            "obs": "free_path", "dev": "none", "event_type": "", "side": "",
            "segment": "", "source_clip": prefix + folder.name,
        })
    return jobs, skipped


# ---------------------------------------------------------------------------
# Frame decoding + dataset writing
# ---------------------------------------------------------------------------

def decode_frames(video_path, stride):
    """Return [(frame_index, time_s, bgr_frame), ...] sampled every `stride`."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = []
    idx = -1
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        idx += 1
        if stride > 1 and (idx % stride) != 0:
            continue
        frames.append((idx, (idx / fps) if fps > 0 else 0.0, frame))
    cap.release()
    return frames


def reset_dir(path, overwrite):
    if path.exists():
        if not overwrite:
            raise SystemExit(f"Output exists; pass --overwrite: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


CSV_FIELDS = ["filename", "label", "source", "event_type", "side", "segment",
              "source_clip", "frame_index", "time_s"]


def write_labels_csv(out_dir, rows):
    with (out_dir / "labels.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def write_info(out_dir, name, classes, rows, extra):
    counts = {c: 0 for c in classes}
    src_counts = {}
    for r in rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
        src_counts[r["source"]] = src_counts.get(r["source"], 0) + 1
    payload = {
        "dataset": name,
        "task": extra["task"],
        "classes": classes,
        "total_images": len(rows),
        "class_counts": counts,
        "source_counts": src_counts,
        "source_clips": extra["source_clips"],
        "frame_selection": extra["frame_selection"],
        "label_map": extra["label_map"],
        "sources": extra["sources"],
    }
    (out_dir / "dataset_info.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cuts", default=CONFIG["cuts"],
                    help="Labelled deviation cut folders (cut_info.json with event_type).")
    ap.add_argument("--endpoints", default=CONFIG["endpoints"],
                    help="dataset_endpoints head/tail cut folders (free path).")
    ap.add_argument("--clear", default=CONFIG["clear"],
                    help="Clear clip folders, one sub-folder per clip (free path).")
    ap.add_argument("--clear-date", default=CONFIG["clear_date"],
                    help="DD-MM-YYYY collection date prefixed onto the clear clips' "
                         "source_clip so the trainer's day split can place them "
                         "(default 22-06-2026 = the test day).")
    ap.add_argument("--out", default=CONFIG["out"], help="Output root for the datasets.")
    ap.add_argument("--img-size", type=int, default=CONFIG["img_size"],
                    help="Resize frames to SIZExSIZE (default 224; 0 = native).")
    ap.add_argument("--stride", type=int, default=CONFIG["stride"],
                    help="Keep every Nth frame (default 1 = all frames).")
    ap.add_argument("--jpeg-quality", type=int, default=CONFIG["jpeg_quality"])
    ap.add_argument("--limit", type=int, default=CONFIG["limit"],
                    help="Process only N clips per source (debug).")
    ap.add_argument("--overwrite", action="store_true", default=CONFIG["overwrite"])
    # In a notebook (Kaggle/Colab) the kernel injects args like `-f kernel.json`;
    # ignore unknown args and fall back to CONFIG defaults instead of erroring.
    in_notebook = any("ipykernel" in m or "colab" in m for m in sys.modules)
    if in_notebook:
        args, _ = ap.parse_known_args([])
    else:
        args, _ = ap.parse_known_args()

    out_root = Path(args.out).resolve()
    jobs, skipped = [], []
    finders = (
        ("cuts", lambda r: find_cut_jobs(r, "cuts")),
        ("endpoints", lambda r: find_cut_jobs(r, "endpoints", allow_missing_info=True)),
        ("clear", lambda r: find_clear_jobs(r, clear_date=args.clear_date)),
    )
    for source, finder in finders:
        raw = getattr(args, source)
        if not raw:
            continue
        root = Path(raw).resolve()
        if not root.is_dir():
            raise SystemExit(f"--{source} not found: {root}")
        found, skip = finder(root)
        if args.limit:
            found = found[: args.limit]
        skipped.extend(skip)
        jobs.extend(found)
        print(f"[{source}] {len(found)} clip folders under {root}")
    if not jobs:
        raise SystemExit("No input clips found.")

    ds = {
        "obs": out_root / "obstacle_freepath",
        "dev": out_root / "deviation_lrn",
    }
    for d in ds.values():
        reset_dir(d, args.overwrite)
    for c in OBSTACLE_CLASSES:
        (ds["obs"] / c).mkdir(exist_ok=True)
    for c in DEVIATION_CLASSES:
        (ds["dev"] / c).mkdir(exist_ok=True)

    rows = {k: [] for k in ds}
    jpeg = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
    used_clips = 0

    size = int(args.img_size or 0)

    def emit(key, label, base, frame, job, fidx, time_s):
        if size > 0 and (frame.shape[0] != size or frame.shape[1] != size):
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        fname = f"{base}__f{fidx:05d}.jpg"
        rel = f"{label}/{fname}"
        cv2.imwrite(str(ds[key] / rel), frame, jpeg)
        rows[key].append({
            "filename": rel, "label": label, "source": job["source"],
            "event_type": job["event_type"], "side": job["side"],
            "segment": job["segment"], "source_clip": job["source_clip"],
            "frame_index": fidx, "time_s": round(time_s, 4),
        })

    for n, job in enumerate(jobs, start=1):
        clip_name = job["folder"].name
        frames = decode_frames(job["video"], args.stride)
        if not frames:
            skipped.append({"clip": clip_name, "source": job["source"],
                            "reason": "no_frames"})
            print(f"[{n:04d}/{len(jobs):04d}] skip {clip_name}: no decodable frames")
            continue
        base = f"{job['source']}__{slugify(clip_name)}"
        for fidx, time_s, frame in frames:
            emit("obs", job["obs"], base, frame, job, fidx, time_s)
            emit("dev", job["dev"], base, frame, job, fidx, time_s)
        used_clips += 1
        print(f"[{n:04d}/{len(jobs):04d}] {job['source']:9s} {clip_name} "
              f"-> obs={job['obs']} dev={job['dev']} | {len(frames)} frames")

    obs_map = {"free_path": "no obstacle (livre / endpoints / clear)",
               "obstacle": "obstacle (desvio/parada)"}
    dev_map = {"left": "desvio LEFT", "right": "desvio RIGHT",
               "none": "no deviation (livre/parada/endpoints/clear)"}
    sources = {
        "cuts": str(args.cuts), "endpoints": str(args.endpoints),
        "clear": str(args.clear),
    }
    sel = (f"all frames (stride={args.stride}, "
           f"{'%dx%d' % (size, size) if size > 0 else 'native'} px)")
    summary = {}
    for key, name, classes, task, lmap in (
            ("obs", "obstacle_freepath", OBSTACLE_CLASSES, "obstacle-vs-free-path", obs_map),
            ("dev", "deviation_lrn", DEVIATION_CLASSES, "deviation-left-right-none", dev_map)):
        write_labels_csv(ds[key], rows[key])
        counts = write_info(ds[key], name, classes, rows[key], {
            "task": task, "source_clips": used_clips,
            "frame_selection": sel, "label_map": lmap, "sources": sources,
        })
        summary[name] = {"images": len(rows[key]), "counts": counts}

    print("\n=== Done ===")
    print(f"Clips used: {used_clips}, skipped: {len(skipped)}")
    for name, s in summary.items():
        print(f"  {name}: {s['images']} images  {json.dumps(s['counts'])}")
    print(f"Output root: {out_root}")
    if skipped:
        (out_root / "skipped.json").write_text(
            json.dumps(skipped, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    rc = main()
    # In a notebook `raise SystemExit` prints an ugly traceback; only exit the
    # process when run as a real script.
    if not any("ipykernel" in m or "colab" in m for m in sys.modules):
        raise SystemExit(rc)
