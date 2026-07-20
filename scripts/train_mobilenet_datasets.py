# -*- coding: utf-8 -*-
"""Train MobileNetV1 (alpha=0.5, 128x128, ImageNet weights) on each image dataset
produced by make_kaggle_image_datasets.py.

For every dataset folder (an ImageFolder tree with a labels.csv: columns
`filename,label,event_type,side,source_clip,frame_index,time_s`) it:

  * splits **by clip** (the `source_clip` column) into train/val/test, so frames
    from one clip never land in two splits -> no temporal leakage. The split is
    stratified by the clip's class, so every split sees every class.
  * balances classes with inverse-frequency sample weights computed on the train
    frames (handles e.g. free_path << obstacle).
  * builds MobileNetV1 alpha=0.5 @128 with ImageNet weights (include_top=False,
    global-avg pool) + dropout + a softmax head sized to the dataset's classes.
  * trains in two phases: frozen-base warmup, then low-LR fine-tune of the whole
    net. Best model by val_loss is checkpointed.
  * evaluates on the held-out test clips and writes a confusion matrix + report.

Augmentation note: horizontal flip is auto-DISABLED for the deviation dataset
(classes left/right) because a flip would swap the label; it stays ON for
obstacle/free_path.

Kaggle/Colab: paste into a cell and run (edit CONFIG below), or run as a script:

  python train_mobilenet_datasets.py \
      --datasets-root /kaggle/working/datasets \
      --out-root /kaggle/working/models --epochs 40
"""
import argparse
import csv
import json
import re
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
    "datasets_root": "/kaggle/working/datasets",   # holds the dataset sub-folders
    "datasets": "",                                 # "" = all sub-folders with labels.csv
    "out_root": "/kaggle/working/models",
    "img_size": 128,
    "alpha": 0.5,
    "epochs": 40,
    "warmup_epochs": 8,
    "batch": 32,
    "val_frac": 0.15,        # fraction of the NON-test clips held out for val
    "test_day": 22,          # clips from this collection day-of-month are the test set
    "seed": 42,
}


# ---------------------------------------------------------------------------
# Data loading + clip-grouped, stratified split
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


# ---------------------------------------------------------------------------
# tf.data pipeline + model
# ---------------------------------------------------------------------------

def make_dataset(tf, rows, label_to_idx, weights, img_size, batch, training, hflip, seed):
    paths = [r["path"] for r in rows]
    ys = np.array([label_to_idx[r["label"]] for r in rows], np.int64)
    sw = np.array([weights[int(y)] for y in ys], np.float32)

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
            img = tf.image.random_brightness(img, 25.5)        # ~0.1 on a 0-255 scale
            img = tf.image.random_contrast(img, 0.9, 1.1)
            img = tf.clip_by_value(img, 0.0, 255.0)
        return img, y, w

    return ds.map(load, num_parallel_calls=AUTOTUNE).batch(batch).prefetch(AUTOTUNE)


def build_model(tf, img_size, alpha, n_classes):
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
    h = tf.keras.layers.Dropout(0.2)(base.output)
    out = tf.keras.layers.Dense(n_classes, activation="softmax", name="cls")(h)
    return tf.keras.Model(inputs, out), base


def compile_model(tf, model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        weighted_metrics=["accuracy"],
    )


# ---------------------------------------------------------------------------
# Train one dataset
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

    y_train = [label_to_idx[r["label"]] for r in train]
    weights = inverse_freq_weights(y_train, n_classes)
    print("  sample weights:", {classes[k]: round(v, 2) for k, v in weights.items()})

    train_ds = make_dataset(tf, train, label_to_idx, weights, args.img_size,
                            args.batch, True, hflip, args.seed)
    val_ds = make_dataset(tf, val, label_to_idx, weights, args.img_size,
                          args.batch, False, hflip, args.seed)
    test_ds = make_dataset(tf, test, label_to_idx, weights, args.img_size,
                           args.batch, False, hflip, args.seed)

    model, base = build_model(tf, args.img_size, args.alpha, n_classes)
    out_dir = Path(out_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.keras"
    ckpt = tf.keras.callbacks.ModelCheckpoint(str(model_path), monitor="val_loss",
                                              save_best_only=True, verbose=0)
    early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                             restore_best_weights=True)

    base.trainable = False
    compile_model(tf, model, 1e-3)
    print("\n== Phase 1: warmup (frozen base) ==")
    h1 = model.fit(train_ds, validation_data=val_ds,
                   epochs=args.warmup_epochs, callbacks=[ckpt, early], verbose=2)

    base.trainable = True
    compile_model(tf, model, 1e-4)
    print("\n== Phase 2: fine-tune (whole net, low LR) ==")
    h2 = model.fit(train_ds, validation_data=val_ds, initial_epoch=len(h1.epoch),
                   epochs=args.epochs, callbacks=[ckpt, early], verbose=2)

    model.save(model_path)

    # Test evaluation + confusion matrix.
    print("\n== Test (held-out clips) ==")
    test_metrics = model.evaluate(test_ds, return_dict=True, verbose=0)
    y_true = np.array([label_to_idx[r["label"]] for r in test], np.int64)
    y_prob = model.predict(test_ds, verbose=0)
    y_pred = y_prob.argmax(axis=1)
    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=n_classes).numpy()
    per_class = {}
    for i, c in enumerate(classes):
        tp = int(cm[i, i]); fp = int(cm[:, i].sum() - tp); fn = int(cm[i, :].sum() - tp)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[c] = {"precision": round(prec, 4), "recall": round(rec, 4),
                        "f1": round(f1, 4), "support": int(cm[i, :].sum())}

    print(f"test metrics: {json.dumps({k: round(float(v),4) for k,v in test_metrics.items()})}")
    print("confusion matrix (rows=true, cols=pred):")
    print("    " + "  ".join(f"{c[:8]:>8}" for c in classes))
    for i, c in enumerate(classes):
        print(f"{c[:8]:>8}  " + "  ".join(f"{int(cm[i,j]):8d}" for j in range(n_classes)))
    for c, m in per_class.items():
        print(f"  {c:>10}: P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f} (n={m['support']})")

    report = {
        "dataset": name, "classes": classes, "n_frames": len(rows),
        "img_size": args.img_size, "alpha": args.alpha, "test_day": args.test_day,
        "split": {"train": len(train), "val": len(val), "test": len(test), **clip_stats},
        "sample_weights": {classes[k]: v for k, v in weights.items()},
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    hist = {**h1.history, **{k: h1.history.get(k, []) + v for k, v in h2.history.items()}}
    (out_dir / "history.json").write_text(
        json.dumps({k: [float(x) for x in v] for k, v in hist.items()}, indent=2))
    print(f"saved -> {model_path}")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
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
    in_notebook = any("ipykernel" in m or "colab" in m for m in sys.modules)
    args, _ = ap.parse_known_args([] if in_notebook else None)

    import tensorflow as tf  # noqa: PLC0415
    tf.keras.utils.set_random_seed(args.seed)

    root = Path(args.datasets_root)
    if args.datasets.strip():
        dirs = [root / n.strip() for n in args.datasets.split(",") if n.strip()]
    else:
        dirs = sorted(p for p in root.iterdir() if (p / "labels.csv").is_file())
    if not dirs:
        raise SystemExit(f"No datasets with labels.csv under {root}")
    print(f"Training on {len(dirs)} dataset(s): {[d.name for d in dirs]}")

    summary = {}
    for d in dirs:
        rep = train_one(tf, d, args.out_root, args)
        if rep:
            summary[d.name] = {"test_metrics": rep["test_metrics"],
                               "per_class": rep["per_class"]}

    print(f"\n{'='*70}\nALL DONE\n{'='*70}")
    for name, s in summary.items():
        acc = s["test_metrics"].get("accuracy")
        print(f"  {name}: test_acc={acc:.4f}" if acc is not None else f"  {name}: {s['test_metrics']}")
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    (Path(args.out_root) / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    rc = main()
    if not any("ipykernel" in m or "colab" in m for m in sys.modules):
        raise SystemExit(rc)
