# -*- coding: utf-8 -*-
"""Build labelled image datasets for Kaggle from the `deviation` cut folders.

Self-contained: it needs **only** the cut folders (each a `cut_info.json` + the
already-trimmed `<clip>.mp4`), so it runs on Kaggle from the uploaded
`datasetcortado/deviation` directory with nothing but OpenCV (+ optional PyTorch
for the DINOv2 variants). It does NOT import `sensecv.app` or any IMU data.

It produces four ImageFolder-style datasets (class sub-folders + a `labels.csv`):

  1. obstacle_freepath/        obstacle vs free_path        (all decoded frames)
  2. deviation_lrn/            left / right / none          (all decoded frames)
  3. obstacle_freepath_dino/   obstacle vs free_path        (DINOv2-selected frames)
  4. deviation_lrn_dino/       left / right / none          (DINOv2-selected frames)

Label mapping (project taxonomy, see docs/wiki/concepts/classification-taxonomy.md
and crop-suggestion.md):

  event_type   side    obstacle label   deviation label
  desvio       LEFT     obstacle         left
  desvio       RIGHT    obstacle         right
  parada       NONE     obstacle         none
  livre        NONE     free_path        none

Datasets 1 & 2 share the same underlying frames (organised under two label
schemes); 3 & 4 likewise share the DINOv2-selected subset. Each output folder is
fully self-contained so it can be zipped and uploaded to Kaggle on its own.

Kaggle usage (in a notebook cell, Internet ON so DINOv2 weights can download):

    !python make_kaggle_image_datasets.py \
        --input /kaggle/input/<your-dataset>/deviation \
        --out   /kaggle/working

Local usage:

    python scripts/make_kaggle_image_datasets.py \
        --input data/datasetcortado/datasetcortado/deviation \
        --out   data/derived/kaggle_image_datasets
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
# (when run with `!python make_kaggle_image_datasets.py --input ...`) override
# anything set here.
# ---------------------------------------------------------------------------
CONFIG = {
    "input": "/kaggle/input/datasets/isaqueefraim/datasetandre/deviation",  # folder with the cut sub-folders
    "out": "/kaggle/working/datasets",                   # output root
    "stride": 1,
    "jpeg_quality": 95,
    "no_dino": False,
    "dino_threshold": 0.98,
    "dino_max_per_clip": 8,
    "limit": None,
    "overwrite": True,
}

# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

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
    return None, None  # unknown event_type -> skip


OBSTACLE_CLASSES = ["free_path", "obstacle"]
DEVIATION_CLASSES = ["none", "left", "right"]


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------

def find_cut_dirs(root):
    """Every directory under `root` that holds a cut_info.json and an .mp4."""
    cuts = []
    for ci in sorted(root.rglob("cut_info.json")):
        folder = ci.parent
        mp4s = sorted(folder.glob("*.mp4"))
        if not mp4s:
            print(f"[warn] no .mp4 next to {ci}")
            continue
        try:
            info = json.loads(ci.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] unreadable {ci}: {e}")
            continue
        cuts.append({"folder": folder, "video": mp4s[0], "info": info})
    return cuts


def slugify(name):
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in name)


# ---------------------------------------------------------------------------
# Frame decoding
# ---------------------------------------------------------------------------

def decode_frames(video_path, stride):
    """Return [(frame_index, time_s, bgr_frame), ...] sampled every `stride`."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], 0.0
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
        time_s = (idx / fps) if fps > 0 else 0.0
        frames.append((idx, time_s, frame))
    cap.release()
    return frames, fps


# ---------------------------------------------------------------------------
# DINOv2 frame selection (lazy: only built when --dino is on)
# ---------------------------------------------------------------------------

class Dinov2Selector:
    """Embed frames with DINOv2 ViT-S/14 and keep a diverse, de-duplicated subset.

    A frame is kept when its cosine similarity to the last kept frame drops below
    `threshold` (0.98 — consecutive-frame cosine sits ~0.975-0.985 for this
    footage). The first frame is always kept and `max_per_clip` caps the count.
    """

    def __init__(self, threshold=0.98, max_per_clip=8, batch_size=32):
        import torch  # noqa: PLC0415

        self.torch = torch
        self.threshold = threshold
        self.max_per_clip = max_per_clip
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_model().to(self.device).eval()
        # ImageNet normalisation, 224x224, as DINOv2 expects.
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        print(f"[dino] DINOv2 ViT-S/14 ready on {self.device}")

    def _load_model(self):
        try:
            return self.torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        except Exception as e:  # noqa: BLE001
            print(f"[dino] torch.hub failed ({e}); trying transformers...")
            from transformers import AutoModel  # noqa: PLC0415

            hf = AutoModel.from_pretrained("facebook/dinov2-small")

            class _HFWrap(self.torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m

                def forward(self, x):
                    return self.m(x).pooler_output

            return _HFWrap(hf)

    def _embed(self, frames):
        torch = self.torch
        feats = []
        with torch.no_grad():
            for i in range(0, len(frames), self.batch_size):
                batch = frames[i:i + self.batch_size]
                tens = []
                for f in batch:
                    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
                    tens.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0)
                x = torch.stack(tens).to(self.device)
                x = (x - self.mean) / self.std
                emb = self.model(x)
                emb = torch.nn.functional.normalize(emb, dim=1)
                feats.append(emb.cpu())
        return torch.cat(feats, dim=0)

    def select(self, frames):
        """frames: list of bgr arrays. Returns indices to keep (into `frames`)."""
        if not frames:
            return []
        if len(frames) == 1:
            return [0]
        emb = self._embed(frames)
        kept = [0]
        last = emb[0]
        for i in range(1, len(frames)):
            if len(kept) >= self.max_per_clip:
                break
            cos = float(self.torch.dot(last, emb[i]))
            if cos < self.threshold:
                kept.append(i)
                last = emb[i]
        return kept


# ---------------------------------------------------------------------------
# Dataset writers
# ---------------------------------------------------------------------------

def reset_dir(path, overwrite):
    if path.exists():
        if not overwrite:
            raise SystemExit(f"Output exists; pass --overwrite: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_labels_csv(out_dir, rows):
    headers = ["filename", "label", "event_type", "side",
               "source_clip", "frame_index", "time_s"]
    with (out_dir / "labels.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)


def write_info(out_dir, name, classes, rows, extra):
    counts = {c: 0 for c in classes}
    for r in rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    payload = {
        "dataset": name,
        "task": extra["task"],
        "classes": classes,
        "total_images": len(rows),
        "class_counts": counts,
        "source_clips": extra["source_clips"],
        "frame_selection": extra["frame_selection"],
        "label_map": extra["label_map"],
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
    ap.add_argument("--input", default=CONFIG["input"],
                    help="Folder holding the cut sub-folders (e.g. .../deviation).")
    ap.add_argument("--out", default=CONFIG["out"], help="Output root for the datasets.")
    ap.add_argument("--stride", type=int, default=CONFIG["stride"],
                    help="Keep every Nth frame for the full datasets (default 1 = all).")
    ap.add_argument("--jpeg-quality", type=int, default=CONFIG["jpeg_quality"])
    ap.add_argument("--no-dino", action="store_true", default=CONFIG["no_dino"],
                    help="Skip the two DINOv2-selected datasets.")
    ap.add_argument("--dino-threshold", type=float, default=CONFIG["dino_threshold"],
                    help="Keep a frame when DINOv2 cosine to the last kept < threshold.")
    ap.add_argument("--dino-max-per-clip", type=int, default=CONFIG["dino_max_per_clip"])
    ap.add_argument("--limit", type=int, default=CONFIG["limit"],
                    help="Process only N clips (debug).")
    ap.add_argument("--overwrite", action="store_true", default=CONFIG["overwrite"])
    # In a notebook (Kaggle/Colab) the kernel injects args like `-f kernel.json`;
    # ignore unknown args and fall back to CONFIG defaults instead of erroring.
    in_notebook = any("ipykernel" in m or "colab" in m for m in sys.modules)
    if in_notebook:
        args, _ = ap.parse_known_args([])
    else:
        args, _ = ap.parse_known_args()

    root = Path(args.input).resolve()
    out_root = Path(args.out).resolve()
    if not root.is_dir():
        raise SystemExit(f"Input not found: {root}")

    cuts = find_cut_dirs(root)
    if args.limit:
        cuts = cuts[:args.limit]
    if not cuts:
        raise SystemExit(f"No cut folders (cut_info.json + .mp4) under {root}")
    print(f"Found {len(cuts)} labelled cut folders under {root}")

    # Prepare the four dataset trees.
    ds = {
        "obs_full": out_root / "obstacle_freepath",
        "dev_full": out_root / "deviation_lrn",
    }
    if not args.no_dino:
        ds["obs_dino"] = out_root / "obstacle_freepath_dino"
        ds["dev_dino"] = out_root / "deviation_lrn_dino"
    for d in ds.values():
        reset_dir(d, args.overwrite)
    for d in (ds["obs_full"], ds.get("obs_dino")):
        if d:
            for c in OBSTACLE_CLASSES:
                (d / c).mkdir(exist_ok=True)
    for d in (ds["dev_full"], ds.get("dev_dino")):
        if d:
            for c in DEVIATION_CLASSES:
                (d / c).mkdir(exist_ok=True)

    selector = None if args.no_dino else Dinov2Selector(
        threshold=args.dino_threshold, max_per_clip=args.dino_max_per_clip)

    rows = {k: [] for k in ds}
    jpeg = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
    used_clips = 0
    skipped = []

    def emit(dataset_key, label, base, frame, fidx, time_s, et, sd, clip):
        fname = f"{base}__f{fidx:05d}.jpg"
        rel = f"{label}/{fname}"
        cv2.imwrite(str(ds[dataset_key] / rel), frame, jpeg)
        rows[dataset_key].append({
            "filename": rel, "label": label, "event_type": et, "side": sd,
            "source_clip": clip, "frame_index": fidx, "time_s": round(time_s, 4),
        })

    for n, cut in enumerate(cuts, start=1):
        info = cut["info"]
        et, sd = info.get("event_type"), info.get("side")
        obs_label, dev_label = labels_for(et, sd)
        clip_name = cut["folder"].name
        if obs_label is None:
            skipped.append({"clip": clip_name, "reason": f"unknown event_type={et}"})
            print(f"[{n:03d}/{len(cuts):03d}] skip {clip_name}: unknown event_type={et}")
            continue

        frames, fps = decode_frames(cut["video"], args.stride)
        if not frames:
            skipped.append({"clip": clip_name, "reason": "no_frames"})
            print(f"[{n:03d}/{len(cuts):03d}] skip {clip_name}: no decodable frames")
            continue

        base = slugify(clip_name)
        # Full datasets: every (strided) frame.
        for fidx, time_s, frame in frames:
            emit("obs_full", obs_label, base, frame, fidx, time_s, et, sd, clip_name)
            emit("dev_full", dev_label, base, frame, fidx, time_s, et, sd, clip_name)

        # DINOv2 datasets: diverse subset.
        n_dino = 0
        if selector is not None:
            keep = selector.select([f for _, _, f in frames])
            for j in keep:
                fidx, time_s, frame = frames[j]
                emit("obs_dino", obs_label, base, frame, fidx, time_s, et, sd, clip_name)
                emit("dev_dino", dev_label, base, frame, fidx, time_s, et, sd, clip_name)
            n_dino = len(keep)

        used_clips += 1
        print(f"[{n:03d}/{len(cuts):03d}] {clip_name} [{et}/{sd}] "
              f"-> obs={obs_label} dev={dev_label} | {len(frames)} frames"
              + (f", {n_dino} dino" if selector is not None else ""))

    # Write labels + info per dataset.
    obs_map = {"free_path": "no obstacle (livre)", "obstacle": "obstacle (desvio/parada)"}
    dev_map = {"left": "desvio LEFT", "right": "desvio RIGHT",
               "none": "no deviation (livre/parada)"}
    full_sel = f"all frames (stride={args.stride})"
    dino_sel = (f"DINOv2 ViT-S/14 cosine dedup (threshold={args.dino_threshold}, "
                f"max {args.dino_max_per_clip}/clip)")

    summary = {}
    specs = [
        ("obs_full", "obstacle_freepath", OBSTACLE_CLASSES, "obstacle-vs-free-path", obs_map, full_sel),
        ("dev_full", "deviation_lrn", DEVIATION_CLASSES, "deviation-left-right-none", dev_map, full_sel),
    ]
    if selector is not None:
        specs += [
            ("obs_dino", "obstacle_freepath_dino", OBSTACLE_CLASSES, "obstacle-vs-free-path", obs_map, dino_sel),
            ("dev_dino", "deviation_lrn_dino", DEVIATION_CLASSES, "deviation-left-right-none", dev_map, dino_sel),
        ]
    for key, name, classes, task, lmap, sel in specs:
        write_labels_csv(ds[key], rows[key])
        counts = write_info(ds[key], name, classes, rows[key], {
            "task": task, "source_clips": used_clips,
            "frame_selection": sel, "label_map": lmap,
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
    # In a notebook (Kaggle/Colab) `raise SystemExit` prints an ugly traceback;
    # only exit the process when run as a real script.
    if not any("ipykernel" in m or "colab" in m for m in sys.modules):
        raise SystemExit(rc)
