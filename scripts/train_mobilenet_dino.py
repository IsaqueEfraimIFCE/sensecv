# -*- coding: utf-8 -*-
"""Standalone DINO pipeline: BUILD DINOv2-frame-selected datasets from the
already-built all-frames IMAGE datasets, then TRAIN MobileNetV1 (alpha=0.5,
ImageNet weights) on each (`obstacle_freepath_dino`, `deviation_lrn_dino`).

Self-contained — no dependency on make_kaggle_allframes_dataset.py or
train_mobilenet_datasets.py; paste this single file into a Kaggle/Colab cell
and run (Internet ON so the DINOv2 weights can download). Input is the
uploaded `kaggle_allframes_datasets_224` root (the two ImageFolder trees
`obstacle_freepath/` + `deviation_lrn/`, each with a labels.csv), so no video
decoding is needed — just OpenCV, PyTorch (DINOv2) and TensorFlow (training).

BUILD (image-dataset variant of make_kaggle_image_datasets.py's dino step):
  * reads `obstacle_freepath/labels.csv` from --input and groups the frames by
    physical sequence (the filename base before `__fNNNNN.jpg`, which keeps
    head/tail endpoint cuts separate), ordered by frame_index.
  * embeds each sequence's frames with DINOv2 ViT-S/14 (torch.hub
    facebookresearch/dinov2, transformers facebook/dinov2-small fallback) and
    keeps a diverse subset — a frame is kept when its cosine similarity to the
    last kept frame drops below --dino-threshold (0.98), capped at
    --dino-max-per-clip (8).
  * copies the kept frames (both label schemes, matched by filename) into
    `obstacle_freepath_dino/` + `deviation_lrn_dino/` under --datasets-root,
    each with a filtered labels.csv (same columns as the source, including
    `source`/`segment`) and a dataset_info.json. Skipped when both labels.csv
    already exist (or --no-build / --rebuild to force either way).

TRAIN, per dataset:
  * splits **by clip** (the `source_clip` column) into train/val/test, so frames
    from one clip never land in two splits -> no temporal leakage. test = the
    fixed day-of-month (--test-day, default 22); train/val = the rest, stratified.
  * runs several reproducible fine-tuning strategies. The default sweep compares
    the original baseline, conservative last-block tuning, and clip+class-balanced
    regularized tuning. A strategy is selected by validation macro-F1 only.
  * builds MobileNetV1 alpha=0.5 @224 (the dataset's native size) with ImageNet
    weights + an embedded Rescaling(1/127.5,-1) layer (raw 0-255 in -> drop-in
    usable in the SenseCV viewer), dropout, and a softmax head in canonical
    class order.
  * trains two phases (frozen warmup -> low-LR fine-tune), restores the global
    best val-loss checkpoint (instead of accidentally saving the final epoch),
    and writes per-run validation/test confusion matrices and reports.
  * promotes the validation-selected run to <dataset>/model.keras. Test results
    are reported for diagnosis but are never used to choose the promoted model.

Augmentation note: horizontal flip is auto-DISABLED for the deviation dataset
(classes left/right) because a flip would swap the label; it stays ON for
obstacle/free_path.

Kaggle/Colab: paste into a cell and run (edit CONFIG below), or run as a script:

  python train_mobilenet_dino.py \
      --input /kaggle/input/datasets/isaqueefraim/dataset8/kaggle_allframes_datasets_224 \
      --datasets-root /kaggle/working/datasets \
      --out-root /kaggle/working/models_dino --epochs 40
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

_DATE_RE = re.compile(r"(\d{2})-(\d{2})-(\d{4})")


def day_of(clip_name):
    """Day-of-month parsed from a clip name like SenseCV-22-06-2026-UFC_..."""
    m = _DATE_RE.search(clip_name or "")
    return int(m.group(1)) if m else None

# ---------------------------------------------------------------------------
# Notebook config (edit + run the cell). CLI flags override these.
# ---------------------------------------------------------------------------
CONFIG = {
    # -- build (all-frames image datasets -> the two _dino datasets) --
    "input": "/kaggle/input/datasets/isaqueefraim/dataset8/kaggle_allframes_datasets_224",
    "no_build": False,       # skip the build entirely (datasets already exist)
    "rebuild": False,        # force a rebuild even if labels.csv already exist
    "dino_threshold": 0.98,
    "dino_max_per_clip": 8,
    "limit": None,           # process only N sequences (debug)
    # -- train --
    "datasets_root": "/kaggle/working/datasets",   # holds the dataset sub-folders
    "datasets": "obstacle_freepath_dino,deviation_lrn_dino",
    "out_root": "/kaggle/working/models_dino",
    "img_size": 224,         # the all-frames dataset's native size
    "alpha": 0.5,
    "epochs": 40,
    "warmup_epochs": 8,
    "batch": 32,
    "val_frac": 0.15,        # fraction of the NON-test clips held out for val
    "test_day": 22,          # clips from this collection day-of-month are the test set
    "seed": 42,
    # baseline = original recipe; last_block = safer transfer learning;
    # clip_balanced = equal class/clip influence + stronger label-safe augmentation.
    "trials": "baseline,last_block,clip_balanced",
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

# input scheme folder -> output _dino folder
SCHEMES = {"obstacle_freepath": "obstacle_freepath_dino",
           "deviation_lrn": "deviation_lrn_dino"}
_FRAME_SUFFIX_RE = re.compile(r"__f\d{5}\.jpg$")


def sequence_of(filename):
    """`label/<base>__fNNNNN.jpg` -> `<base>` (the physical frame sequence).

    The base carries the source prefix and any `_head`/`_tail` suffix, so each
    contiguous video's frames group together while endpoint cuts of the same
    clip stay separate sequences (they are non-contiguous footage).
    """
    return _FRAME_SUFFIX_RE.sub("", Path(filename).name)


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
# BUILD: all-frames image datasets -> the two _dino datasets
# ---------------------------------------------------------------------------

def read_scheme_csv(scheme_dir):
    """Rows of a scheme's labels.csv (list of dicts) + its column order."""
    csv_path = scheme_dir / "labels.csv"
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def build_dino_datasets(args):
    """DINO-select frames from the all-frames image datasets under --input."""
    import cv2  # noqa: PLC0415

    in_root = Path(args.input).resolve()
    out_root = Path(args.datasets_root).resolve()
    schemes = {}
    for src_name in SCHEMES:
        d = in_root / src_name
        if not (d / "labels.csv").is_file():
            raise SystemExit(f"Input scheme not found: {d / 'labels.csv'}")
        schemes[src_name] = d

    # Selection runs ONCE on the obstacle scheme's frames; the deviation scheme
    # shares the same underlying images (same `<base>__fNNNNN.jpg` names under
    # different label folders), so kept filenames transfer 1:1.
    obs_rows, obs_fields = read_scheme_csv(schemes["obstacle_freepath"])
    dev_rows, dev_fields = read_scheme_csv(schemes["deviation_lrn"])
    dev_by_name = {Path(r["filename"]).name: r for r in dev_rows}

    seqs = defaultdict(list)
    for r in obs_rows:
        seqs[sequence_of(r["filename"])].append(r)
    seq_names = sorted(seqs)
    print(f"Found {len(obs_rows)} frames in {len(seq_names)} sequences under {in_root}")
    if args.limit:
        seq_names = seq_names[:args.limit]
        print(f"[debug] limited to the first {len(seq_names)} sequences")

    out_dirs = {}
    for src_name, out_name in SCHEMES.items():
        d = out_root / out_name
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        out_dirs[src_name] = d

    selector = Dinov2Selector(
        cv2, threshold=args.dino_threshold, max_per_clip=args.dino_max_per_clip)

    kept_rows = {"obstacle_freepath": [], "deviation_lrn": []}
    skipped = []

    def copy_kept(src_name, row):
        src = schemes[src_name] / row["filename"]
        dst = out_dirs[src_name] / row["filename"]
        dst.parent.mkdir(exist_ok=True)
        shutil.copy2(src, dst)
        kept_rows[src_name].append(row)

    for n, seq in enumerate(seq_names, start=1):
        rows = sorted(seqs[seq], key=lambda r: int(r["frame_index"]))
        frames, valid_rows = [], []
        for r in rows:
            img = cv2.imread(str(schemes["obstacle_freepath"] / r["filename"]))
            if img is None:
                skipped.append({"file": r["filename"], "reason": "unreadable"})
                continue
            frames.append(img)
            valid_rows.append(r)
        if not frames:
            skipped.append({"sequence": seq, "reason": "no_frames"})
            print(f"[{n:04d}/{len(seq_names):04d}] skip {seq}: no readable frames")
            continue

        keep = selector.select(frames)
        for j in keep:
            r = valid_rows[j]
            copy_kept("obstacle_freepath", r)
            dev_r = dev_by_name.get(Path(r["filename"]).name)
            if dev_r is not None:
                copy_kept("deviation_lrn", dev_r)
            else:
                skipped.append({"file": r["filename"],
                                "reason": "no deviation_lrn counterpart"})
        print(f"[{n:04d}/{len(seq_names):04d}] {seq} "
              f"-> {len(keep)}/{len(frames)} frames kept")

    dino_sel = (f"DINOv2 ViT-S/14 cosine dedup (threshold={args.dino_threshold}, "
                f"max {args.dino_max_per_clip}/sequence) over {in_root.name}")
    for src_name, fields in (("obstacle_freepath", obs_fields),
                             ("deviation_lrn", dev_fields)):
        out_dir = out_dirs[src_name]
        rows = kept_rows[src_name]
        with (out_dir / "labels.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        counts = Counter(r["label"] for r in rows)
        info = {}
        info_path = schemes[src_name] / "dataset_info.json"
        if info_path.is_file():
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                info = {}
        info.update({
            "dataset": SCHEMES[src_name],
            "total_images": len(rows),
            "class_counts": dict(counts),
            "frame_selection": dino_sel,
            "source_dataset": str(schemes[src_name]),
        })
        (out_dir / "dataset_info.json").write_text(
            json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  {SCHEMES[src_name]}: {len(rows)} images  {json.dumps(dict(counts))}")

    print(f"Build done: {len(seq_names)} sequences, {len(skipped)} skipped items "
          f"-> {out_root}")
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


def build_model(tf, img_size, alpha, n_classes, dropout):
    # Embedded preprocessing: raw 0-255 RGB -> [-1, 1] (MobileNet's range). Baking
    # it into the graph means the SenseCV viewer can feed raw pixels (the same
    # contract as the original two-head model's Rescaling(1/127.5,-1) layer).
    inputs = tf.keras.Input((img_size, img_size, 3))
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)
    try:
        base = tf.keras.applications.MobileNet(
            input_tensor=x, alpha=alpha,
            include_top=False, weights="imagenet", pooling="avg")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] ImageNet weights unavailable ({e}); training from scratch")
        base = tf.keras.applications.MobileNet(
            input_tensor=x, alpha=alpha,
            include_top=False, weights=None, pooling="avg")
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


# ---------------------------------------------------------------------------
# TRAIN: one dataset
# ---------------------------------------------------------------------------

def train_one(tf, dataset_dir, out_root, args):
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
            tf, args.img_size, args.alpha, n_classes, cfg["dropout"])
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
            "alpha": args.alpha, "test_day": args.test_day,
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
                    help="Root of the all-frames image datasets "
                         "(holds obstacle_freepath/ + deviation_lrn/).")
    ap.add_argument("--no-build", action="store_true", default=CONFIG["no_build"],
                    help="Skip the dataset build; train on existing datasets only.")
    ap.add_argument("--rebuild", action="store_true", default=CONFIG["rebuild"],
                    help="Rebuild the _dino datasets even if they already exist.")
    ap.add_argument("--dino-threshold", type=float, default=CONFIG["dino_threshold"],
                    help="Keep a frame when DINOv2 cosine to the last kept < threshold.")
    ap.add_argument("--dino-max-per-clip", type=int, default=CONFIG["dino_max_per_clip"])
    ap.add_argument("--limit", type=int, default=CONFIG["limit"],
                    help="Process only N sequences in the build (debug).")
    # train flags
    ap.add_argument("--datasets-root", default=CONFIG["datasets_root"])
    ap.add_argument("--datasets", default=CONFIG["datasets"],
                    help="comma-separated dataset folder names; empty = all with labels.csv")
    ap.add_argument("--out-root", default=CONFIG["out_root"])
    ap.add_argument("--img-size", type=int, default=CONFIG["img_size"])
    ap.add_argument("--alpha", type=float, default=CONFIG["alpha"])
    ap.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    ap.add_argument("--warmup-epochs", type=int, default=CONFIG["warmup_epochs"])
    ap.add_argument("--batch", type=int, default=CONFIG["batch"])
    ap.add_argument("--val-frac", type=float, default=CONFIG["val_frac"],
                    help="fraction of the NON-test clips held out for validation")
    ap.add_argument("--test-day", type=int, default=CONFIG["test_day"],
                    help="collection day-of-month whose clips form the test set (e.g. 22)")
    ap.add_argument("--seed", type=int, default=CONFIG["seed"])
    ap.add_argument("--trials", default=CONFIG["trials"],
                    help="comma-separated strategies: " + ",".join(TRIALS))
    in_notebook = any("ipykernel" in m or "colab" in m for m in sys.modules)
    args, _ = ap.parse_known_args([] if in_notebook else None)

    root = Path(args.datasets_root)

    # -- BUILD: all-frames images -> the two _dino datasets (skipped when built) --
    dino_dirs = [root / out_name for out_name in SCHEMES.values()]
    have_all = all((d / "labels.csv").is_file() for d in dino_dirs)
    if args.no_build:
        print("[build] skipped (--no-build)")
    elif have_all and not args.rebuild:
        print(f"[build] skipped: datasets already exist under {root} "
              f"(pass --rebuild to redo)")
    else:
        build_dino_datasets(args)

    # -- TRAIN --
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
