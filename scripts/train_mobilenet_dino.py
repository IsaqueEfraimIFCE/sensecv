# -*- coding: utf-8 -*-
"""Build DINOv2-selected datasets from SenseCV deviation cuts, then train a
selectable MobileNetV1/V2/V3/V4 backbone with pretrained ImageNet weights on
`obstacle_freepath_dino` and `deviation_lrn_dino`.

Self-contained for Kaggle/Colab: the input is the uploaded
`Todos_cortes_auto_trim/deviation` folder containing one labelled, trimmed MP4
per child folder. Labels come from `cut_info.json`, with the root `sources.csv`
as fallback. Internet must be ON for DINOv2 and ImageNet weight downloads.

BUILD:
  * maps desvio LEFT/RIGHT -> obstacle + left/right, parada NONE -> obstacle +
    none, and livre NONE -> free_path + none.
  * decodes each cut video and embeds its frames with DINOv2 ViT-S/14 (torch.hub
    facebookresearch/dinov2, transformers facebook/dinov2-small fallback) and
    keeps a diverse subset — a frame is kept when its cosine similarity to the
    last kept frame drops below --dino-threshold (0.96 by default), capped at
    --dino-max-per-clip (25 by default).
  * writes the shared selected frame subset into
    `obstacle_freepath_dino/` + `deviation_lrn_dino/` under --datasets-root,
    each with labels.csv and dataset_info.json. The build is skipped when both
    outputs already exist (or controlled with --no-build / --rebuild).
  * additionally runs every `*sem_obstaculos*.jpg` from --day22-clear-dir
    through the same per-clip DINOv2 selector, then writes the selected subset
    as free_path/none while preserving day-22 provenance from labels.csv.

TRAIN, per dataset:
  * splits **by clip** (the `source_clip` column) into train/val/test, so frames
    from one clip never land in two splits -> no temporal leakage. test = the
    fixed day-of-month (--test-day, default 22); train/val = the rest, stratified.
  * runs several reproducible fine-tuning strategies. The default sweep compares
    the original baseline, conservative last-block tuning, and clip+class-balanced
    regularized tuning. A strategy is selected by validation macro-F1 only.
  * selects the backbone with --mobilenet-version (v1, v2, v3, or v4). Every
    option is the baseline-width (alpha=1.0) ImageNet model at 224x224; other
    alpha/resolution values are rejected so pretrained-weight compatibility is
    explicit rather than silently falling back to random weights.
  * V1/V2/V3 use TensorFlow/Keras and save model.keras with embedded [-1, 1]
    preprocessing (raw 0-255 input, drop-in usable in the SenseCV viewer).
    V4 uses timm/PyTorch's 224px Conv-Small preset and saves model.pt because
    tf.keras.applications does not provide MobileNetV4.
  * trains two phases (frozen warmup -> low-LR fine-tune), restores the global
    best val-loss checkpoint (instead of accidentally saving the final epoch),
    and writes per-run validation/test confusion matrices and reports.
  * promotes the validation-selected run to <dataset>/model.keras (V1-V3) or
    <dataset>/model.pt (V4). Test results are reported for diagnosis but are
    never used to choose the promoted model.

Augmentation note: horizontal flip is auto-DISABLED for the deviation dataset
(classes left/right) because a flip would swap the label; it stays ON for
obstacle/free_path.

Kaggle/Colab: paste into a cell and run (edit CONFIG below), or run as a script:

  python train_mobilenet_dino.py \
      --input /kaggle/input/datasets/isaqueefraim/datasetandrefinal/Todos_cortes_auto_trim/deviation \
      --datasets-root /kaggle/working/datasets \
      --out-root /kaggle/working/models_dino \
      --mobilenet-version v2 --epochs 40
"""
import argparse
import csv
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DATASET_BUILDER_CONTRACT = "sensecv-deviation-mp4-day22-clear-v1"
_DATE_RE = re.compile(r"(\d{2})-(\d{2})-(\d{4})")


def day_of(clip_name):
    """Day-of-month parsed from a clip name like SenseCV-22-06-2026-UFC_..."""
    m = _DATE_RE.search(clip_name or "")
    return int(m.group(1)) if m else None

# ---------------------------------------------------------------------------
# Notebook config (edit + run the cell). CLI flags override these.
# ---------------------------------------------------------------------------
CONFIG = {
    # -- build (labelled cut videos -> the two _dino datasets) --
    "input": "/kaggle/input/datasets/isaqueefraim/datasetandrefinal/Todos_cortes_auto_trim/deviation",
    "no_build": False,       # skip the build entirely (datasets already exist)
    "rebuild": False,        # force a rebuild even if labels.csv already exist
    "stride": 1,             # decode every Nth frame before DINO selection
    "jpeg_quality": 95,
    "day22_clear_dir": "/kaggle/input/datasets/isaqueefraim/dataset8/"
                         "kaggle_allframes_datasets_224/deviation_lrn/none",
    "dino_threshold": 0.96,
    "dino_max_per_clip": 25,
    "limit": None,           # process only N sequences (debug)
    # -- train --
    "datasets_root": "/kaggle/working/datasets",   # holds the dataset sub-folders
    "datasets": "obstacle_freepath_dino,deviation_lrn_dino",
    "out_root": "/kaggle/working/models_dino",
    "mobilenet_version": "v1",  # v1 | v2 | v3 | v4
    "img_size": 224,
    "alpha": 1.0,
    "epochs": 40,
    "warmup_epochs": 8,
    "batch": 32,
    "val_frac": 0.15,        # fraction of the NON-test clips held out for val
    # The extra sem_obstaculos images above supply day-22 free_path/none while
    # the labelled cuts supply its obstacle/left/right examples.
    "test_day": 22,
    "seed": 43,
    # baseline = original recipe; last_block = safer transfer learning;
    # clip_balanced = equal class/clip influence + stronger label-safe augmentation.
    "trials": "clip_balanced",
}


# Deliberately small sweep: these are different training strategies, not a large
# test-set-driven hyperparameter search. Add a strategy here and name it in
# CONFIG["trials"] / --trials to include it.
TRIALS = {
    "baseline": {
        "dropout": 0.20, "warmup_lr": 1e-3, "finetune_lr": 1e-4,
        "unfreeze_last": 0, "freeze_batchnorm": False,
        "optimizer": "adam", "weight_decay": 0.0,
        "weighting": "inverse_frequency", "augmentation": "basic",
        "patience": 8,
    },
    "last_block": {
        "dropout": 0.35, "warmup_lr": 7e-4, "finetune_lr": 3e-5,
        "unfreeze_last": 30, "freeze_batchnorm": True,
        "optimizer": "adamw", "weight_decay": 1e-5,
        "weighting": "inverse_frequency", "augmentation": "strong",
        "patience": 7,
    },
    "clip_balanced": {
        "dropout": 0.30, "warmup_lr": 7e-4, "finetune_lr": 5e-5,
        "unfreeze_last": 45, "freeze_batchnorm": True,
        "optimizer": "adamw", "weight_decay": 2e-5,
        "weighting": "clip_class", "augmentation": "strong",
        "patience": 8,
    },
}

SCHEMES = {"obstacle_freepath": "obstacle_freepath_dino",
           "deviation_lrn": "deviation_lrn_dino"}
OBSTACLE_CLASSES = ["free_path", "obstacle"]
DEVIATION_CLASSES = ["left", "right", "none"]
MOBILENET_VERSIONS = ("v1", "v2", "v3", "v4")
MOBILENET_V4_TIMM_MODEL = "mobilenetv4_conv_small.e2400_r224_in1k"


def parse_mobilenet_version(value):
    """Accept concise v1..v4 and explicit mobilenetv1..mobilenetv4 names."""
    normalized = str(value).strip().lower().replace("-", "").replace("_", "")
    if normalized.startswith("mobilenet"):
        normalized = normalized[len("mobilenet"):]
    if normalized in MOBILENET_VERSIONS:
        return normalized
    raise argparse.ArgumentTypeError(
        "choose v1, v2, v3, v4 (or mobilenetv1 ... mobilenetv4)")


def labels_for(event_type, side):
    """Map the documented SenseCV taxonomy to both single-head targets."""
    event = (event_type or "").strip().lower()
    direction = (side or "").strip().upper()
    if event == "desvio":
        if direction == "LEFT":
            return "obstacle", "left"
        if direction == "RIGHT":
            return "obstacle", "right"
        return "obstacle", "none"
    if event == "parada":
        return "obstacle", "none"
    if event == "livre":
        return "free_path", "none"
    return None, None


def slugify(name):
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in name)


def read_sources_ledger(root):
    """Return output_folder -> ledger row for cut_info.json fallback labels."""
    path = root / "sources.csv"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {r.get("output_folder", "").strip(): r for r in csv.DictReader(f)}


def find_labelled_cuts(root):
    """Discover child cut folders, with sources.csv as the label fallback."""
    ledger = read_sources_ledger(root)
    cuts = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        videos = sorted(folder.glob("*.mp4"))
        if not videos:
            continue
        info = None
        info_path = folder / "cut_info.json"
        if info_path.is_file():
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] unreadable {info_path}: {exc}; trying sources.csv")
        label_source = "cut_info.json"
        if not info:
            info = ledger.get(folder.name)
            label_source = "sources.csv"
        if not info:
            print(f"[warn] no label metadata for {folder.name}")
            continue
        cuts.append({"folder": folder, "video": videos[0], "info": info,
                     "label_source": label_source})
    return cuts


def decode_frames(cv2, video_path, stride):
    """Decode [(frame_index, time_s, BGR frame)] from a trimmed cut."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], 0.0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = []
    index = -1
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        index += 1
        if stride > 1 and index % stride:
            continue
        frames.append((index, index / fps if fps > 0 else 0.0, frame))
    cap.release()
    return frames, fps


def read_day22_clear_images(clear_dir):
    """Read every sem_obstaculos JPG plus its all-frames CSV provenance."""
    clear_dir = Path(clear_dir) if clear_dir else None
    if not clear_dir or not clear_dir.is_dir():
        if clear_dir:
            print(f"[warn] day-22 clear directory not found: {clear_dir}")
        return []

    metadata = {}
    labels_path = clear_dir.parent / "labels.csv"
    if labels_path.is_file():
        with labels_path.open(encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                basename = Path(row.get("filename", "")).name
                if "sem_obstaculos" in basename.lower():
                    metadata[basename] = row
    else:
        print(f"[warn] no labels.csv beside day-22 clear images: {labels_path}; "
              "inferring clip/frame metadata from filenames")

    result = []
    for image_path in sorted(clear_dir.glob("*.jpg")):
        if "sem_obstaculos" not in image_path.name.lower():
            continue
        row = metadata.get(image_path.name, {})
        match = re.search(r"__f(\d+)$", image_path.stem)
        frame_index = int(row.get("frame_index") or (match.group(1) if match else 0))
        sequence = re.sub(r"__f\d+$", "", image_path.stem).replace("__", "_")
        source_clip = (row.get("source_clip") or
                       f"SenseCV-22-06-2026-{sequence}")
        # Enforce day 22 even if an incomplete/mismatched source CSV is mounted.
        if day_of(source_clip) != 22:
            source_clip = f"SenseCV-22-06-2026-{sequence}"
        result.append({
            "path": image_path, "filename": image_path.name,
            "source_clip": source_clip, "frame_index": frame_index,
            "time_s": row.get("time_s") or "",
        })
    return result


# ---------------------------------------------------------------------------
# BUILD: DINOv2 frame selection
# ---------------------------------------------------------------------------

class Dinov2Selector:
    """Embed frames with DINOv2 ViT-S/14 and keep a diverse, de-duplicated subset.

    A frame is kept when its cosine similarity to the last kept frame drops below
    `threshold` (0.98 — consecutive-frame cosine sits ~0.975-0.985 for this
    footage). The first frame is always kept and `max_per_clip` caps the count.
    """

    def __init__(self, cv2, threshold=0.98, max_per_clip=8, batch_size=32):
        import torch  # noqa: PLC0415

        self.cv2 = cv2
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
        torch, cv2 = self.torch, self.cv2
        feats = []
        with torch.no_grad():
            for i in range(0, len(frames), self.batch_size):
                batch = frames[i:i + self.batch_size]
                tens = []
                for f in batch:
                    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    if rgb.shape[:2] != (224, 224):
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
# BUILD: labelled cut videos -> the two _dino datasets
# ---------------------------------------------------------------------------

def build_dino_datasets(args):
    """Decode cut videos and write the shared DINO-selected frame subset."""
    import cv2  # noqa: PLC0415

    in_root = Path(args.input).resolve()
    out_root = Path(args.datasets_root).resolve()
    if not in_root.is_dir():
        raise SystemExit(f"Input deviation folder not found: {in_root}")
    cuts = find_labelled_cuts(in_root)
    if args.limit:
        cuts = cuts[:args.limit]
        print(f"[debug] limited to the first {len(cuts)} clips")
    if not cuts:
        raise SystemExit(f"No labelled cut folders with MP4 files under {in_root}")
    print(f"Found {len(cuts)} labelled cut folders under {in_root}")

    out_dirs = {name: out_root / output for name, output in SCHEMES.items()}
    for name, out_dir in out_dirs.items():
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        classes = OBSTACLE_CLASSES if name == "obstacle_freepath" else DEVIATION_CLASSES
        for class_name in classes:
            (out_dir / class_name).mkdir(exist_ok=True)

    selector = Dinov2Selector(
        cv2, threshold=args.dino_threshold, max_per_clip=args.dino_max_per_clip)
    kept_rows = {name: [] for name in SCHEMES}
    skipped = []
    label_sources = Counter()
    jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
    fields = ["filename", "label", "event_type", "side", "source_clip",
              "frame_index", "time_s", "label_source"]

    def emit(scheme, label, base, frame, frame_index, time_s, event, side,
             clip_name, label_source):
        filename = f"{base}__f{frame_index:05d}.jpg"
        relative = f"{label}/{filename}"
        destination = out_dirs[scheme] / relative
        if not cv2.imwrite(str(destination), frame, jpeg_params):
            raise RuntimeError(f"Could not write frame: {destination}")
        kept_rows[scheme].append({
            "filename": relative,
            "label": label,
            "event_type": event,
            "side": side,
            "source_clip": clip_name,
            "frame_index": frame_index,
            "time_s": round(time_s, 4),
            "label_source": label_source,
        })

    def copy_clear(scheme, label, clear):
        relative = f"{label}/{clear['filename']}"
        destination = out_dirs[scheme] / relative
        shutil.copy2(clear["path"], destination)
        kept_rows[scheme].append({
            "filename": relative,
            "label": label,
            "event_type": "livre",
            "side": "NONE",
            "source_clip": clear["source_clip"],
            "frame_index": clear["frame_index"],
            "time_s": clear["time_s"],
            "label_source": "day22_sem_obstaculos",
        })

    for number, cut in enumerate(cuts, start=1):
        info = cut["info"]
        event = info.get("event_type")
        side = info.get("side")
        obstacle_label, deviation_label = labels_for(event, side)
        clip_name = cut["folder"].name
        if obstacle_label is None:
            skipped.append({"clip": clip_name, "reason": f"unknown event_type={event}"})
            print(f"[{number:04d}/{len(cuts):04d}] skip {clip_name}: unknown label")
            continue

        frames, _ = decode_frames(cv2, cut["video"], args.stride)
        if not frames:
            skipped.append({"clip": clip_name, "reason": "no decodable frames"})
            print(f"[{number:04d}/{len(cuts):04d}] skip {clip_name}: no frames")
            continue
        keep = selector.select([frame for _, _, frame in frames])
        base = slugify(clip_name)
        for selected_index in keep:
            frame_index, time_s, frame = frames[selected_index]
            emit("obstacle_freepath", obstacle_label, base, frame, frame_index,
                 time_s, event, side, clip_name, cut["label_source"])
            emit("deviation_lrn", deviation_label, base, frame, frame_index,
                 time_s, event, side, clip_name, cut["label_source"])
        label_sources[cut["label_source"]] += 1
        print(f"[{number:04d}/{len(cuts):04d}] {clip_name} [{event}/{side}] "
              f"-> {len(keep)}/{len(frames)} frames kept")

    day22_clear_source = read_day22_clear_images(args.day22_clear_dir)
    clear_by_clip = defaultdict(list)
    for clear in day22_clear_source:
        clear_by_clip[clear["source_clip"]].append(clear)
    day22_clear = []
    unreadable_clear = 0
    for clip_name, clip_rows in sorted(clear_by_clip.items()):
        clip_rows.sort(key=lambda row: row["frame_index"])
        frames = []
        valid_rows = []
        for clear in clip_rows:
            frame = cv2.imread(str(clear["path"]))
            if frame is None:
                unreadable_clear += 1
                continue
            frames.append(frame)
            valid_rows.append(clear)
        if frames:
            day22_clear.extend(valid_rows[index] for index in selector.select(frames))

    for clear in day22_clear:
        copy_clear("obstacle_freepath", "free_path", clear)
        copy_clear("deviation_lrn", "none", clear)
    if day22_clear_source:
        label_sources["day22_sem_obstaculos"] += len(day22_clear)
        print(f"DINO-selected {len(day22_clear)}/{len(day22_clear_source)} "
              f"sem_obstaculos images from {len(clear_by_clip)} day-22 clear "
              f"clips (free_path / none; {unreadable_clear} unreadable)")

    selection = (f"DINOv2 ViT-S/14 cosine dedup (threshold={args.dino_threshold}, "
                 f"max {args.dino_max_per_clip}/clip, decode stride={args.stride})")
    for scheme, out_dir in out_dirs.items():
        rows = kept_rows[scheme]
        with (out_dir / "labels.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        counts = Counter(row["label"] for row in rows)
        classes = OBSTACLE_CLASSES if scheme == "obstacle_freepath" else DEVIATION_CLASSES
        info = {
            "builder_contract": DATASET_BUILDER_CONTRACT,
            "dataset": SCHEMES[scheme],
            "task": ("obstacle-vs-free-path" if scheme == "obstacle_freepath"
                     else "deviation-left-right-none"),
            "classes": classes,
            "total_images": len(rows),
            "class_counts": {name: counts.get(name, 0) for name in classes},
            "source_clips": len({row["source_clip"] for row in rows}),
            "source_dataset": str(in_root),
            "label_sources": dict(label_sources),
            "day22_clear_source_images": len(day22_clear_source),
            "day22_clear_selected_images": len(day22_clear),
            "day22_clear_source_clips": len(clear_by_clip),
            "day22_clear_unreadable_images": unreadable_clear,
            "day22_clear_dino_selection": {
                "threshold": args.dino_threshold,
                "max_per_clip": args.dino_max_per_clip,
            },
            "frame_selection": selection,
        }
        (out_dir / "dataset_info.json").write_text(
            json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  {SCHEMES[scheme]}: {len(rows)} images  {json.dumps(dict(counts))}")

    print(f"Build done: {len(cuts)} clips, {len(skipped)} skipped -> {out_root}")
    if skipped:
        (out_root / "skipped_dino.json").write_text(
            json.dumps(skipped, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# TRAIN: data loading + clip-grouped, stratified split
# ---------------------------------------------------------------------------

def read_labels_csv(dataset_dir):
    """Return rows [{path, label, clip}] from a dataset's labels.csv."""
    csv_path = dataset_dir / "labels.csv"
    rows = []
    with csv_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = dataset_dir / r["filename"]
            if not p.is_file():
                continue
            rows.append({"path": str(p), "label": r["label"],
                         "clip": r.get("source_clip") or Path(r["filename"]).stem})
    return rows


def split_by_clip(rows, val_frac, test_day, seed):
    """Split over CLIPS (not frames). Each clip -> exactly one split, no leakage.

    * test  = every clip whose collection day-of-month == test_day (imbalance is
              fine here, it is the fixed held-out day).
    * train/val = the remaining clips, split stratified by class so every class
              appears in both (val_frac of each class's non-test clips -> val).
    """
    rng = np.random.default_rng(seed)
    # clip -> label (clips are single-label here; take the majority just in case)
    clip_label = {}
    clip_rows = defaultdict(list)
    for r in rows:
        clip_rows[r["clip"]].append(r)
    for clip, rs in clip_rows.items():
        clip_label[clip] = Counter(r["label"] for r in rs).most_common(1)[0][0]

    test_clips = {c for c in clip_label if day_of(c) == test_day}

    by_class = defaultdict(list)
    for clip, lab in clip_label.items():
        if clip not in test_clips:
            by_class[lab].append(clip)

    train_clips, val_clips = set(), set()
    for lab, clips in by_class.items():
        clips = list(clips)
        rng.shuffle(clips)
        n = len(clips)
        n_val = max(1, int(round(n * val_frac))) if n > 2 else (1 if n > 1 else 0)
        n_val = min(n_val, max(0, n - 1))  # leave >=1 for train
        val_clips.update(clips[:n_val])
        train_clips.update(clips[n_val:])

    def collect(clip_set):
        return [r for r in rows if r["clip"] in clip_set]

    return collect(train_clips), collect(val_clips), collect(test_clips), {
        "train_clips": len(train_clips), "val_clips": len(val_clips),
        "test_clips": len(test_clips),
    }


def inverse_freq_weights(int_labels, n_classes):
    counts = Counter(int_labels)
    total = len(int_labels)
    return {c: (total / (n_classes * counts[c]) if counts.get(c) else 0.0)
            for c in range(n_classes)}


def training_weights(rows, label_to_idx, mode):
    """Return one weight per row plus a compact, JSON-safe description.

    clip_class gives every class equal total influence, then gives every clip
    containing that class equal influence inside the class. This prevents a
    long/highly redundant sequence from outweighing several short clips.
    """
    ys = [label_to_idx[r["label"]] for r in rows]
    n_classes = len(label_to_idx)
    if mode == "inverse_frequency":
        by_class = inverse_freq_weights(ys, n_classes)
        values = np.asarray([by_class[y] for y in ys], np.float32)
        return values, {"mode": mode, "by_class": by_class}
    if mode != "clip_class":
        raise ValueError(f"Unknown weighting mode: {mode}")

    groups = Counter((label_to_idx[r["label"]], r["clip"]) for r in rows)
    clips_per_class = Counter(y for y, _clip in groups)
    total = len(rows)
    values = np.asarray([
        total / (n_classes * clips_per_class[label_to_idx[r["label"]]] *
                 groups[(label_to_idx[r["label"]], r["clip"])])
        for r in rows
    ], np.float32)
    return values, {
        "mode": mode,
        "clips_per_class": dict(clips_per_class),
        "min": float(values.min()), "max": float(values.max()),
        "mean": float(values.mean()),
    }


# ---------------------------------------------------------------------------
# TRAIN: tf.data pipeline + model
# ---------------------------------------------------------------------------

def make_dataset(tf, rows, label_to_idx, sample_weights, img_size, batch,
                 training, hflip, seed, augmentation="none"):
    paths = [r["path"] for r in rows]
    ys = np.array([label_to_idx[r["label"]] for r in rows], np.int64)
    sw = (np.ones(len(rows), np.float32) if sample_weights is None
          else np.asarray(sample_weights, np.float32))

    AUTOTUNE = tf.data.AUTOTUNE
    ds = tf.data.Dataset.from_tensor_slices((tf.constant(paths), ys, sw))
    if training:
        ds = ds.shuffle(min(len(paths), 4096), seed=seed, reshuffle_each_iteration=True)

    def load(path, y, w):
        # Feed RAW 0-255 float pixels; normalisation lives INSIDE the model (a
        # Rescaling layer, see build_model) so the saved .keras is drop-in usable
        # in the SenseCV viewer, which feeds raw 0-255 (app.py _sensemodel_preprocess).
        img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)  # RGB
        img = tf.image.resize(img, (img_size, img_size))
        img = tf.cast(img, tf.float32)
        if training:
            if hflip:
                img = tf.image.random_flip_left_right(img)
            if augmentation == "strong":
                # Label-safe for left/right: mild zoom/crop and photometric
                # jitter, but never a horizontal flip for deviation classes.
                pad = max(4, int(round(img_size * 0.05)))
                img = tf.image.resize(img, (img_size + 2 * pad, img_size + 2 * pad))
                img = tf.image.random_crop(img, (img_size, img_size, 3), seed=seed)
                img = tf.image.random_brightness(img, 38.25)   # 0.15 on 0-255
                img = tf.image.random_contrast(img, 0.80, 1.20)
                img = tf.image.random_saturation(img, 0.85, 1.15)
            elif augmentation == "basic":
                img = tf.image.random_brightness(img, 25.5)    # 0.10 on 0-255
                img = tf.image.random_contrast(img, 0.90, 1.10)
            img = tf.clip_by_value(img, 0.0, 255.0)
        return img, y, w

    return ds.map(load, num_parallel_calls=AUTOTUNE).batch(batch).prefetch(AUTOTUNE)


def build_model(tf, version, img_size, alpha, n_classes, dropout):
    # Embedded preprocessing: raw 0-255 RGB -> [-1, 1] (MobileNet's range). Baking
    # it into the graph means the SenseCV viewer can feed raw pixels (the same
    # contract as the original two-head model's Rescaling(1/127.5,-1) layer).
    inputs = tf.keras.Input((img_size, img_size, 3))
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)
    constructors = {
        "v1": tf.keras.applications.MobileNet,
        "v2": tf.keras.applications.MobileNetV2,
        # Large is the baseline-width MobileNetV3 variant. Preprocessing is
        # disabled because the outer Rescaling layer already maps to [-1, 1].
        "v3": tf.keras.applications.MobileNetV3Large,
    }
    if version not in constructors:
        raise ValueError(f"TensorFlow backbone not available for {version}")
    kwargs = {
        "input_tensor": x, "alpha": alpha, "include_top": False,
        "weights": "imagenet", "pooling": "avg",
    }
    if version == "v3":
        kwargs["include_preprocessing"] = False
    try:
        base = constructors[version](**kwargs)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not load pretrained ImageNet weights for MobileNet{version.upper()} "
            f"alpha={alpha} at {img_size}x{img_size}. Training from scratch is "
            f"intentionally disabled: {exc}") from exc
    h = tf.keras.layers.Dropout(dropout)(base.output)
    out = tf.keras.layers.Dense(n_classes, activation="softmax", name="cls")(h)
    return tf.keras.Model(inputs, out), base


def compile_model(tf, model, lr, optimizer_name="adam", weight_decay=0.0):
    if optimizer_name == "adamw":
        try:
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=lr, weight_decay=weight_decay, clipnorm=1.0)
        except AttributeError:
            print("[warn] AdamW unavailable; falling back to Adam")
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


def configure_finetuning(tf, base, unfreeze_last, freeze_batchnorm):
    """Unfreeze all base layers (0) or only the last N, optionally keeping BN fixed."""
    base.trainable = True
    cutoff = 0 if not unfreeze_last else max(0, len(base.layers) - unfreeze_last)
    trainable = 0
    for i, layer in enumerate(base.layers):
        enabled = i >= cutoff
        if freeze_batchnorm and isinstance(layer, tf.keras.layers.BatchNormalization):
            enabled = False
        layer.trainable = enabled
        trainable += int(enabled)
    return trainable


def classification_report(tf, model, ds, rows, label_to_idx, classes):
    y_true = np.asarray([label_to_idx[r["label"]] for r in rows], np.int64)
    y_prob = model.predict(ds, verbose=0)
    y_pred = y_prob.argmax(axis=1)
    cm = tf.math.confusion_matrix(
        y_true, y_pred, num_classes=len(classes)).numpy()
    per_class = {}
    for i, name in enumerate(classes):
        tp = int(cm[i, i]); fp = int(cm[:, i].sum() - tp); fn = int(cm[i, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "support": int(cm[i, :].sum()),
        }
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(np.mean([m["f1"] for m in per_class.values()])),
        "balanced_accuracy": float(np.mean([m["recall"] for m in per_class.values()])),
        "confusion_matrix": cm.tolist(), "per_class": per_class,
    }


def report_from_predictions(y_true, y_pred, classes):
    """Framework-independent classification report used by the PyTorch path."""
    y_true = np.asarray(y_true, np.int64)
    y_pred = np.asarray(y_pred, np.int64)
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for truth, prediction in zip(y_true, y_pred):
        cm[truth, prediction] += 1
    per_class = {}
    for i, name in enumerate(classes):
        tp = int(cm[i, i]); fp = int(cm[:, i].sum() - tp); fn = int(cm[i, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "support": int(cm[i, :].sum()),
        }
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(np.mean([m["f1"] for m in per_class.values()])),
        "balanced_accuracy": float(np.mean([m["recall"] for m in per_class.values()])),
        "confusion_matrix": cm.tolist(), "per_class": per_class,
    }


def train_one_torch(dataset_dir, out_root, args):
    """Train MobileNetV4 with timm while preserving the split/report contract."""
    try:
        import torch
        import torch.nn.functional as functional
        import timm
        from PIL import Image
        from timm.data import resolve_model_data_config
        from torchvision import transforms
    except ImportError as exc:
        raise SystemExit(
            "MobileNetV4 requires PyTorch/timm. On Kaggle run: "
            "!pip install -q timm") from exc

    name = dataset_dir.name
    print(f"\n{'='*70}\nDATASET: {name} (PyTorch/timm)\n{'='*70}")
    rows = read_labels_csv(dataset_dir)
    if not rows:
        print(f"[skip] no resolvable images in {dataset_dir}")
        return None
    present = {row["label"] for row in rows}
    classes = next((order for order in
                    (["free_path", "obstacle"], ["left", "right", "none"])
                    if present == set(order)), sorted(present))
    label_to_idx = {label: i for i, label in enumerate(classes)}
    hflip = not ({"left", "right"} & set(classes))
    train, val, test, clip_stats = split_by_clip(
        rows, args.val_frac, args.test_day, args.seed)
    print(f"classes={classes}  n={len(rows)} frames  hflip={hflip}")
    for split_name, split_rows in (("train", train), ("val", val), ("test", test)):
        print(f"  {split_name}: {len(split_rows)} frames | "
              f"{dict(Counter(r['label'] for r in split_rows))}")
        if not split_rows:
            print(f"[skip] split '{split_name}' empty for {name}")
            return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device={device}  pretrained={MOBILENET_V4_TIMM_MODEL}")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    class FrameDataset(torch.utils.data.Dataset):
        def __init__(self, source_rows, transform, weights=None):
            self.rows = source_rows
            self.transform = transform
            self.weights = (np.ones(len(source_rows), np.float32) if weights is None
                            else np.asarray(weights, np.float32))

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            with Image.open(row["path"]) as image:
                tensor = self.transform(image.convert("RGB"))
            return (tensor, label_to_idx[row["label"]], float(self.weights[index]))

    def make_model(dropout):
        try:
            return timm.create_model(
                MOBILENET_V4_TIMM_MODEL, pretrained=True,
                num_classes=len(classes), drop_rate=dropout)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Could not load pretrained timm weights for "
                f"{MOBILENET_V4_TIMM_MODEL}. Training from scratch is disabled: "
                f"{exc}") from exc

    # Resolve the exact normalization stored with the pretrained timm preset.
    probe = make_model(0.0)
    data_cfg = resolve_model_data_config(probe)
    del probe
    normalize = transforms.Normalize(data_cfg["mean"], data_cfg["std"])
    eval_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(), normalize,
    ])

    def training_transform(kind):
        operations = []
        if kind == "strong":
            operations.append(transforms.RandomResizedCrop(
                args.img_size, scale=(0.90, 1.0), ratio=(0.95, 1.05)))
        else:
            operations.append(transforms.Resize((args.img_size, args.img_size)))
        if hflip:
            operations.append(transforms.RandomHorizontalFlip())
        jitter = 0.15 if kind == "strong" else 0.10
        operations.extend([
            transforms.ColorJitter(
                brightness=jitter, contrast=jitter,
                saturation=(jitter if kind == "strong" else 0.0)),
            transforms.ToTensor(), normalize,
        ])
        return transforms.Compose(operations)

    def loader(source_rows, transform, weights=None, shuffle=False):
        generator = torch.Generator().manual_seed(args.seed)
        return torch.utils.data.DataLoader(
            FrameDataset(source_rows, transform, weights), batch_size=args.batch,
            # Keep zero workers: this file is commonly pasted into a notebook
            # cell and the local Dataset class is not spawn-picklable there.
            shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available(),
            generator=generator)

    val_loader = loader(val, eval_transform)
    test_loader = loader(test, eval_transform)

    def evaluate(model, data_loader):
        model.eval()
        total_loss = 0.0; total = 0; truth = []; predictions = []
        with torch.no_grad():
            for images, labels, _weights in data_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                total_loss += float(functional.cross_entropy(
                    logits, labels, reduction="sum").item())
                total += labels.numel()
                truth.extend(labels.cpu().tolist())
                predictions.extend(logits.argmax(dim=1).cpu().tolist())
        report = report_from_predictions(truth, predictions, classes)
        metrics = {"loss": total_loss / max(1, total), **{
            key: report[key] for key in
            ("accuracy", "macro_f1", "balanced_accuracy")}}
        return metrics, report

    out_dir = Path(out_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    trial_names = [item.strip() for item in args.trials.split(",") if item.strip()]
    unknown = [item for item in trial_names if item not in TRIALS]
    if unknown or not trial_names:
        raise SystemExit(f"Unknown/empty --trials {unknown}; choose from {sorted(TRIALS)}")
    trial_reports = {}

    for trial_name in trial_names:
        cfg = dict(TRIALS[trial_name])
        print(f"\n{'-'*70}\nTRIAL: {trial_name}\n{json.dumps(cfg, indent=2)}\n{'-'*70}")
        model = make_model(cfg["dropout"]).to(device)
        sample_weights, weight_info = training_weights(
            train, label_to_idx, cfg["weighting"])
        train_loader = loader(
            train, training_transform(cfg["augmentation"]), sample_weights,
            shuffle=True)
        trial_dir = out_dir / "trials" / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        model_path = trial_dir / "model.pt"
        history = defaultdict(list)
        best_loss = float("inf")

        def run_phase(start_epoch, end_epoch, lr, warmup):
            nonlocal best_loss
            for parameter in model.parameters():
                parameter.requires_grad = not warmup
            if warmup:
                for parameter in model.get_classifier().parameters():
                    parameter.requires_grad = True
            elif cfg["unfreeze_last"]:
                parameters = list(model.parameters())
                for parameter in parameters:
                    parameter.requires_grad = False
                for parameter in parameters[-cfg["unfreeze_last"]:]:
                    parameter.requires_grad = True
            if cfg["freeze_batchnorm"]:
                for module in model.modules():
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        for parameter in module.parameters():
                            parameter.requires_grad = False
            trainable = [p for p in model.parameters() if p.requires_grad]
            optimizer_cls = torch.optim.AdamW if cfg["optimizer"] == "adamw" else torch.optim.Adam
            optimizer = optimizer_cls(
                trainable, lr=lr,
                weight_decay=(cfg["weight_decay"] if cfg["optimizer"] == "adamw" else 0.0))
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.3, patience=3, min_lr=1e-7)
            bad_epochs = 0
            for epoch in range(start_epoch, end_epoch):
                model.train()
                if cfg["freeze_batchnorm"]:
                    for module in model.modules():
                        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                            module.eval()
                running_loss = 0.0; correct = 0; seen = 0
                for images, labels, weights in train_loader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    weights = weights.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(images)
                    losses = functional.cross_entropy(logits, labels, reduction="none")
                    loss = (losses * weights).mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()
                    running_loss += float(loss.item()) * labels.numel()
                    correct += int((logits.argmax(1) == labels).sum().item())
                    seen += labels.numel()
                val_metrics, _ = evaluate(model, val_loader)
                scheduler.step(val_metrics["loss"])
                history["loss"].append(running_loss / max(1, seen))
                history["accuracy"].append(correct / max(1, seen))
                history["val_loss"].append(val_metrics["loss"])
                history["val_accuracy"].append(val_metrics["accuracy"])
                history["learning_rate"].append(optimizer.param_groups[0]["lr"])
                print(f"epoch {epoch + 1}/{args.epochs} loss={history['loss'][-1]:.4f} "
                      f"acc={history['accuracy'][-1]:.4f} "
                      f"val_loss={val_metrics['loss']:.4f} "
                      f"val_acc={val_metrics['accuracy']:.4f}")
                if val_metrics["loss"] < best_loss:
                    best_loss = val_metrics["loss"]
                    bad_epochs = 0
                    torch.save({
                        "state_dict": model.state_dict(),
                        "model_name": MOBILENET_V4_TIMM_MODEL,
                        "classes": classes, "img_size": args.img_size,
                        "alpha": args.alpha,
                    }, model_path)
                else:
                    bad_epochs += 1
                    if bad_epochs >= cfg["patience"]:
                        print(f"early stopping after {bad_epochs} epochs without val-loss improvement")
                        break

        print("\n== Phase 1: warmup (frozen base) ==")
        run_phase(0, args.warmup_epochs, cfg["warmup_lr"], True)
        print("\n== Phase 2: fine-tune ==")
        run_phase(args.warmup_epochs, args.epochs, cfg["finetune_lr"], False)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        val_metrics, val_report = evaluate(model, val_loader)
        test_metrics, test_report = evaluate(model, test_loader)
        report = {
            "dataset": name, "trial": trial_name, "strategy": cfg,
            "classes": classes, "n_frames": len(rows), "img_size": args.img_size,
            "alpha": args.alpha, "mobilenet_version": "v4",
            "timm_model": MOBILENET_V4_TIMM_MODEL,
            "training_backend": "pytorch", "test_day": args.test_day,
            "split": {"train": len(train), "val": len(val), "test": len(test),
                      **clip_stats},
            "weighting": weight_info, "validation_metrics": val_metrics,
            "validation": val_report, "test_metrics": test_metrics,
            "confusion_matrix": test_report["confusion_matrix"],
            "per_class": test_report["per_class"],
        }
        (trial_dir / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        (trial_dir / "history.json").write_text(
            json.dumps(dict(history), indent=2), encoding="utf-8")
        trial_reports[trial_name] = report
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    selected_name = max(trial_names, key=lambda item: (
        trial_reports[item]["validation_metrics"]["macro_f1"],
        trial_reports[item]["validation_metrics"]["accuracy"],
        -trial_reports[item]["validation_metrics"]["loss"]))
    selected = trial_reports[selected_name]
    promoted_path = out_dir / "model.pt"
    shutil.copy2(out_dir / "trials" / selected_name / "model.pt", promoted_path)
    selected["selected_by"] = "validation_macro_f1"
    selected["selected_trial"] = selected_name
    selected["trials"] = {
        trial: {"validation_metrics": report["validation_metrics"],
                "test_metrics": report["test_metrics"]}
        for trial, report in trial_reports.items()}
    comparison_rows = []
    for trial, report in trial_reports.items():
        row = {"trial": trial, "selected": trial == selected_name}
        row.update({f"val_{k}": v for k, v in report["validation_metrics"].items()})
        row.update({f"test_{k}": v for k, v in report["test_metrics"].items()})
        row.update({f"param_{k}": v for k, v in report["strategy"].items()})
        comparison_rows.append(row)
    fields = list(dict.fromkeys(key for row in comparison_rows for key in row))
    with (out_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader(); writer.writerows(comparison_rows)
    (out_dir / "report.json").write_text(
        json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.copy2(out_dir / "trials" / selected_name / "history.json",
                 out_dir / "history.json")
    print(f"\nselected {selected_name} by validation macro-F1 -> {promoted_path}")
    return selected


# ---------------------------------------------------------------------------
# TRAIN: one dataset
# ---------------------------------------------------------------------------

def train_one(tf, dataset_dir, out_root, args):
    if args.mobilenet_version == "v4":
        return train_one_torch(dataset_dir, out_root, args)
    name = dataset_dir.name
    print(f"\n{'='*70}\nDATASET: {name}\n{'='*70}")
    rows = read_labels_csv(dataset_dir)
    if not rows:
        print(f"[skip] no resolvable images in {dataset_dir}")
        return None

    # Canonical class order so the softmax index matches what the SenseCV webview
    # expects (app.py SENSECV_OBSTACLE_LABELS=['NONE','OBSTACLE'] and
    # SENSECV_DEVIATION_LABELS=['LEFT','RIGHT','NONE']). The webview auto-detects a
    # single-head model by output width (2=obstacle, 3=deviation) and labels by
    # argmax, so the training order below must line up with those vocabularies.
    present = {r["label"] for r in rows}
    classes = None
    for order in (["free_path", "obstacle"], ["left", "right", "none"]):
        if present == set(order):
            classes = order
            break
    if classes is None:
        classes = sorted(present)
    label_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    # Flip is unsafe when the label encodes left/right.
    hflip = not ({"left", "right"} & set(classes))
    print(f"classes={classes}  n={len(rows)} frames  hflip={hflip}")

    train, val, test, clip_stats = split_by_clip(
        rows, args.val_frac, args.test_day, args.seed)
    print(f"  test = day-{args.test_day} clips only (imbalance ignored)")
    for split_name, rs in (("train", train), ("val", val), ("test", test)):
        c = Counter(r["label"] for r in rs)
        print(f"  {split_name}: {len(rs)} frames | {dict(c)}")
        if not rs:
            print(f"[skip] split '{split_name}' empty for {name} "
                  f"(no day-{args.test_day} clips?)")
            return None
    missing_test_classes = sorted(set(classes) - {r["label"] for r in test})
    if missing_test_classes:
        print(f"[warn] test day {args.test_day} has no examples for: "
              f"{missing_test_classes}; per-class test metrics will be incomplete")
    print(f"  clips: {clip_stats}")

    out_dir = Path(out_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    trial_names = [x.strip() for x in args.trials.split(",") if x.strip()]
    unknown = [x for x in trial_names if x not in TRIALS]
    if unknown:
        raise SystemExit(f"Unknown trial(s) {unknown}; choose from {sorted(TRIALS)}")
    if not trial_names:
        raise SystemExit("--trials must name at least one strategy")

    # Validation/test weights are intentionally all 1. Training weights should
    # not redefine the meaning of held-out loss or accuracy.
    val_ds = make_dataset(tf, val, label_to_idx, None, args.img_size,
                          args.batch, False, hflip, args.seed)
    test_ds = make_dataset(tf, test, label_to_idx, None, args.img_size,
                           args.batch, False, hflip, args.seed)
    trial_reports = {}

    for trial_name in trial_names:
        cfg = dict(TRIALS[trial_name])
        print(f"\n{'-'*70}\nTRIAL: {trial_name}\n{json.dumps(cfg, indent=2)}\n{'-'*70}")
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(args.seed)

        sample_weights, weight_info = training_weights(
            train, label_to_idx, cfg["weighting"])
        print("  weights:", json.dumps(weight_info))
        train_ds = make_dataset(
            tf, train, label_to_idx, sample_weights, args.img_size, args.batch,
            True, hflip, args.seed, augmentation=cfg["augmentation"])

        model, base = build_model(
            tf, args.mobilenet_version, args.img_size, args.alpha, n_classes,
            cfg["dropout"])
        trial_dir = out_dir / "trials" / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        model_path = trial_dir / "model.keras"
        ckpt = tf.keras.callbacks.ModelCheckpoint(
            str(model_path), monitor="val_loss", mode="min",
            save_best_only=True, verbose=0)

        def callbacks():
            return [
                ckpt,
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=cfg["patience"], mode="min"),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.3, patience=3,
                    min_lr=1e-7, mode="min", verbose=1),
            ]

        base.trainable = False
        compile_model(tf, model, cfg["warmup_lr"], cfg["optimizer"],
                      cfg["weight_decay"])
        print("\n== Phase 1: warmup (frozen base) ==")
        h1 = model.fit(
            train_ds, validation_data=val_ds, epochs=args.warmup_epochs,
            callbacks=callbacks(), verbose=2)

        n_trainable = configure_finetuning(
            tf, base, cfg["unfreeze_last"], cfg["freeze_batchnorm"])
        compile_model(tf, model, cfg["finetune_lr"], cfg["optimizer"],
                      cfg["weight_decay"])
        scope = "whole base" if not cfg["unfreeze_last"] else f"last {cfg['unfreeze_last']} layers"
        print(f"\n== Phase 2: fine-tune {scope} ({n_trainable} trainable layers) ==")
        h2 = model.fit(
            train_ds, validation_data=val_ds, initial_epoch=len(h1.epoch),
            epochs=args.epochs, callbacks=callbacks(), verbose=2)

        # Critical: reload the global best checkpoint. Saving `model` here would
        # overwrite it with the final epoch, which can be materially worse.
        del model
        model = tf.keras.models.load_model(model_path)
        val_metrics = {k: float(v) for k, v in
                       model.evaluate(val_ds, return_dict=True, verbose=0).items()}
        test_metrics = {k: float(v) for k, v in
                        model.evaluate(test_ds, return_dict=True, verbose=0).items()}
        val_report = classification_report(
            tf, model, val_ds, val, label_to_idx, classes)
        test_report = classification_report(
            tf, model, test_ds, test, label_to_idx, classes)
        val_metrics.update({k: val_report[k] for k in
                            ("accuracy", "macro_f1", "balanced_accuracy")})
        test_metrics.update({k: test_report[k] for k in
                             ("accuracy", "macro_f1", "balanced_accuracy")})

        history = {}
        for key in set(h1.history) | set(h2.history):
            history[key] = ([float(x) for x in h1.history.get(key, [])] +
                            [float(x) for x in h2.history.get(key, [])])
        report = {
            "dataset": name, "trial": trial_name, "strategy": cfg,
            "classes": classes, "n_frames": len(rows), "img_size": args.img_size,
            "alpha": args.alpha, "mobilenet_version": args.mobilenet_version,
            "training_backend": "tensorflow", "test_day": args.test_day,
            "split": {"train": len(train), "val": len(val), "test": len(test),
                      **clip_stats},
            "weighting": weight_info, "validation_metrics": val_metrics,
            "validation": val_report, "test_metrics": test_metrics,
            "confusion_matrix": test_report["confusion_matrix"],
            "per_class": test_report["per_class"],
        }
        (trial_dir / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        (trial_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8")
        trial_reports[trial_name] = report

        print(f"validation: acc={val_metrics['accuracy']:.4f} "
              f"macro_f1={val_metrics['macro_f1']:.4f} "
              f"balanced_acc={val_metrics['balanced_accuracy']:.4f}")
        print(f"test (not used for selection): acc={test_metrics['accuracy']:.4f} "
              f"macro_f1={test_metrics['macro_f1']:.4f} "
              f"balanced_acc={test_metrics['balanced_accuracy']:.4f}")
        for c, metrics in test_report["per_class"].items():
            print(f"  {c:>10}: P={metrics['precision']:.3f} "
                  f"R={metrics['recall']:.3f} F1={metrics['f1']:.3f} "
                  f"(n={metrics['support']})")
        del model

    # Select and promote strictly by validation macro-F1. Accuracy and lower
    # validation loss are deterministic tie-breakers; test metrics are excluded.
    selected_name = max(
        trial_names,
        key=lambda x: (
            trial_reports[x]["validation_metrics"]["macro_f1"],
            trial_reports[x]["validation_metrics"]["accuracy"],
            -trial_reports[x]["validation_metrics"]["loss"],
        ))
    selected = trial_reports[selected_name]
    promoted_path = out_dir / "model.keras"
    shutil.copy2(out_dir / "trials" / selected_name / "model.keras", promoted_path)
    selected["selected_by"] = "validation_macro_f1"
    selected["selected_trial"] = selected_name
    selected["trials"] = {
        trial: {
            "validation_metrics": rep["validation_metrics"],
            "test_metrics": rep["test_metrics"],
        } for trial, rep in trial_reports.items()
    }
    comparison_rows = []
    for trial, rep in trial_reports.items():
        row = {"trial": trial, "selected": trial == selected_name}
        row.update({f"val_{k}": v for k, v in rep["validation_metrics"].items()})
        row.update({f"test_{k}": v for k, v in rep["test_metrics"].items()})
        row.update({f"param_{k}": v for k, v in rep["strategy"].items()})
        comparison_rows.append(row)
    comparison_fields = list(dict.fromkeys(
        key for row in comparison_rows for key in row))
    with (out_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=comparison_fields)
        writer.writeheader()
        writer.writerows(comparison_rows)
    (out_dir / "report.json").write_text(
        json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")
    # Compatibility for tools expecting history.json beside the promoted model.
    shutil.copy2(out_dir / "trials" / selected_name / "history.json",
                 out_dir / "history.json")
    print(f"\nselected {selected_name} by validation macro-F1 "
          f"({selected['validation_metrics']['macro_f1']:.4f}) -> {promoted_path}")
    return selected


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # build flags
    ap.add_argument("--input", default=CONFIG["input"],
                    help="Root Todos_cortes_auto_trim/deviation folder.")
    ap.add_argument("--no-build", action="store_true", default=CONFIG["no_build"],
                    help="Skip the dataset build; train on existing datasets only.")
    ap.add_argument("--rebuild", action="store_true", default=CONFIG["rebuild"],
                    help="Rebuild the _dino datasets even if they already exist.")
    ap.add_argument("--stride", type=int, default=CONFIG["stride"],
                    help="Decode every Nth video frame before DINO selection.")
    ap.add_argument("--jpeg-quality", type=int, default=CONFIG["jpeg_quality"])
    ap.add_argument("--day22-clear-dir", default=CONFIG["day22_clear_dir"],
                    help="deviation_lrn/none folder; *sem_obstaculos*.jpg files "
                         "are DINO-selected per clip as day-22 free_path/none "
                         "(empty disables)")
    ap.add_argument("--dino-threshold", type=float, default=CONFIG["dino_threshold"],
                    help="Keep a frame when DINOv2 cosine to the last kept < threshold.")
    ap.add_argument("--dino-max-per-clip", type=int, default=CONFIG["dino_max_per_clip"])
    ap.add_argument("--limit", type=int, default=CONFIG["limit"],
                    help="Process only N cut clips in the build (debug).")
    # train flags
    ap.add_argument("--datasets-root", default=CONFIG["datasets_root"])
    ap.add_argument("--datasets", default=CONFIG["datasets"],
                    help="comma-separated dataset folder names; empty = all with labels.csv")
    ap.add_argument("--out-root", default=CONFIG["out_root"])
    ap.add_argument("--mobilenet-version", default=CONFIG["mobilenet_version"],
                    type=parse_mobilenet_version,
                    choices=MOBILENET_VERSIONS,
                    help="pretrained backbone: v1/v2/v3 (Keras) or v4 (timm); "
                         "full mobilenetvN names are also accepted")
    ap.add_argument("--img-size", type=int, default=CONFIG["img_size"])
    ap.add_argument("--alpha", type=float, default=CONFIG["alpha"])
    ap.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    ap.add_argument("--warmup-epochs", type=int, default=CONFIG["warmup_epochs"])
    ap.add_argument("--batch", type=int, default=CONFIG["batch"])
    ap.add_argument("--val-frac", type=float, default=CONFIG["val_frac"],
                    help="fraction of the NON-test clips held out for validation")
    ap.add_argument("--test-day", type=int, default=CONFIG["test_day"],
                    help="collection day-of-month whose clips form the test set (default 22)")
    ap.add_argument("--seed", type=int, default=CONFIG["seed"])
    ap.add_argument("--trials", default=CONFIG["trials"],
                    help="comma-separated strategies: " + ",".join(TRIALS))
    in_notebook = any("ipykernel" in m or "colab" in m for m in sys.modules)
    args, _ = ap.parse_known_args([] if in_notebook else None)

    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("--jpeg-quality must be between 1 and 100")
    if args.img_size != 224:
        raise SystemExit("Pretrained MobileNet selection requires --img-size 224")
    if not np.isclose(args.alpha, 1.0):
        raise SystemExit("Pretrained MobileNet selection requires --alpha 1.0")

    root = Path(args.datasets_root)

    # -- BUILD: labelled cut videos -> the two _dino datasets (skipped when built) --
    dino_dirs = [root / out_name for out_name in SCHEMES.values()]
    clear_dir = Path(args.day22_clear_dir) if args.day22_clear_dir else None
    expected_clear = (sum(1 for path in clear_dir.glob("*.jpg")
                          if "sem_obstaculos" in path.name.lower())
                      if clear_dir and clear_dir.is_dir() else 0)

    def dataset_is_current(directory):
        if not (directory / "labels.csv").is_file():
            return False
        info_path = directory / "dataset_info.json"
        if not info_path.is_file():
            return False
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            if info.get("builder_contract") != DATASET_BUILDER_CONTRACT:
                return False
            if not expected_clear:
                return True
            clear_selection = info.get("day22_clear_dino_selection", {})
            return (
                int(info.get("day22_clear_source_images", 0)) == expected_clear
                and int(info.get("day22_clear_selected_images", 0)) > 0
                and int(info.get("day22_clear_unreadable_images", -1)) == 0
                and float(clear_selection.get("threshold", -1)) == args.dino_threshold
                and int(clear_selection.get("max_per_clip", -1)) == args.dino_max_per_clip
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False

    have_all = all(dataset_is_current(directory) for directory in dino_dirs)
    if args.no_build:
        print("[build] skipped (--no-build)")
    elif have_all and not args.rebuild:
        print(f"[build] skipped: datasets already exist under {root} "
              f"(pass --rebuild to redo)")
    else:
        if expected_clear:
            print(f"[build] DINO-selecting {expected_clear} day-22 "
                  "sem_obstaculos source images per clear clip")
        build_dino_datasets(args)

    # -- TRAIN --
    tf = None
    if args.mobilenet_version != "v4":
        import tensorflow as tf  # noqa: PLC0415
        tf.keras.utils.set_random_seed(args.seed)

    if args.datasets.strip():
        dirs = [root / n.strip() for n in args.datasets.split(",") if n.strip()]
    else:
        dirs = sorted(p for p in root.iterdir() if (p / "labels.csv").is_file())
    dirs = [d for d in dirs if (d / "labels.csv").is_file()]
    if not dirs:
        raise SystemExit(f"No datasets with labels.csv under {root} "
                         f"(looked for {args.datasets or 'all'})")
    backend = "PyTorch/timm" if args.mobilenet_version == "v4" else "TensorFlow/Keras"
    print(f"Training MobileNet{args.mobilenet_version.upper()} alpha=1.0 @224 "
          f"with pretrained weights via {backend}")
    print(f"Training on {len(dirs)} dataset(s): {[d.name for d in dirs]}")

    summary = {}
    for d in dirs:
        rep = train_one(tf, d, args.out_root, args)
        if rep:
            summary[d.name] = {"selected_trial": rep["selected_trial"],
                               "selected_by": rep["selected_by"],
                               "validation_metrics": rep["validation_metrics"],
                               "test_metrics": rep["test_metrics"],
                               "trials": rep["trials"],
                               "per_class": rep["per_class"]}

    print(f"\n{'='*70}\nALL DONE\n{'='*70}")
    for name, s in summary.items():
        acc = s["test_metrics"].get("accuracy")
        macro = s["test_metrics"].get("macro_f1")
        print(f"  {name}: selected={s['selected_trial']} test_acc={acc:.4f} "
              f"test_macro_f1={macro:.4f}" if acc is not None and macro is not None
              else f"  {name}: {s['test_metrics']}")
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    (Path(args.out_root) / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    rc = main()
    if not any("ipykernel" in m or "colab" in m for m in sys.modules):
        raise SystemExit(rc)
