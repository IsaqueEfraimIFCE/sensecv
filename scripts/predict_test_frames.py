# -*- coding: utf-8 -*-
"""Run the trained models over their held-out TEST frames and save a browsable
folder of the predictions.

Standalone Kaggle/Colab companion to the trainers (train_mobilenet_dino.py /
train_mobilenet_datasets.py / train_mobilenet_allframes.py): paste into a cell
after training and run. For every dataset folder (labels.csv) that has a
trained model (`<models-root>/<dataset>/model.keras`) it:

  * rebuilds the SAME test set the trainer held out: every frame whose
    `source_clip` collection day-of-month == --test-day (default 22). The test
    split depends only on the day, so it reproduces exactly, no seed needed.
  * loads the model, reads its input size and class order from report.json
    (fallback: canonical order by output width), and predicts every test frame
    (raw 0-255 in — the models embed their own Rescaling layer).
  * writes to `<out>/<dataset>/`:
      - `<pred_label>/` folders holding a COPY of each test frame, filed under
        the label THE MODEL gave it. Correct frames keep `ok__<name>`;
        misclassified ones are renamed `true-<true_label>__<name>` so mistakes
        stand out while browsing.
      - `predictions.csv`: filename, source_clip, true_label, pred_label,
        confidence, one probability column per class, correct.
      - `summary.json`: accuracy + per-true-label breakdown.

Kaggle/Colab: edit CONFIG and run the cell, or run as a script:

  python predict_test_frames.py \
      --datasets-root /kaggle/working/datasets \
      --models-root   /kaggle/working/models_dino \
      --out           /kaggle/working/test_predictions
"""
import argparse
import csv
import json
import re
import shutil
import sys
from collections import Counter
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
    "datasets_root": "/kaggle/working/datasets",
    # comma-separated dataset folder names; empty = every folder that has both
    # a labels.csv and a trained model under models_root.
    "datasets": "obstacle_freepath_dino,deviation_lrn_dino",
    "models_root": "/kaggle/working/models_dino",
    "out": "/kaggle/working/test_predictions",
    "test_day": 22,          # same held-out day the trainer used
    "batch": 64,
}

CANONICAL_ORDERS = {2: ["free_path", "obstacle"], 3: ["left", "right", "none"]}


def read_test_rows(dataset_dir, test_day):
    """Test rows [{path, filename, label, clip}] = clips of the held-out day."""
    rows = []
    with (dataset_dir / "labels.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = dataset_dir / r["filename"]
            if not p.is_file():
                continue
            clip = r.get("source_clip") or Path(r["filename"]).stem
            if day_of(clip) != test_day:
                continue
            rows.append({"path": str(p), "filename": r["filename"],
                         "label": r["label"], "clip": clip})
    return rows


def model_classes(model, model_dir):
    """Class order the model was trained with (report.json, else canonical)."""
    report = model_dir / "report.json"
    if report.is_file():
        try:
            classes = json.loads(report.read_text(encoding="utf-8"))["classes"]
            if len(classes) == int(model.output_shape[-1]):
                return classes
        except Exception as e:  # noqa: BLE001
            print(f"[warn] unreadable {report}: {e}")
    n_out = int(model.output_shape[-1])
    classes = CANONICAL_ORDERS.get(n_out) or [f"class_{i}" for i in range(n_out)]
    print(f"[warn] no report.json class order; assuming {classes}")
    return classes


def predict_all(tf, model, rows, img_size, batch):
    """Probabilities (n_rows, n_classes) feeding raw 0-255 pixels."""
    paths = tf.constant([r["path"] for r in rows])

    def load(path):
        img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)  # RGB
        img = tf.image.resize(img, (img_size, img_size))
        return tf.cast(img, tf.float32)

    ds = (tf.data.Dataset.from_tensor_slices(paths)
          .map(load, num_parallel_calls=tf.data.AUTOTUNE)
          .batch(batch).prefetch(tf.data.AUTOTUNE))
    return model.predict(ds, verbose=0)


def predict_one(tf, dataset_dir, model_dir, out_root, args):
    name = dataset_dir.name
    print(f"\n{'='*70}\nDATASET: {name}\n{'='*70}")
    model_path = model_dir / "model.keras"
    rows = read_test_rows(dataset_dir, args.test_day)
    if not rows:
        print(f"[skip] no day-{args.test_day} test frames in {dataset_dir}")
        return None
    print(f"  {len(rows)} test frames from "
          f"{len({r['clip'] for r in rows})} day-{args.test_day} clips")

    model = tf.keras.models.load_model(model_path)
    img_size = int(model.input_shape[1])
    classes = model_classes(model, model_dir)
    print(f"  model={model_path}  input={img_size}x{img_size}  classes={classes}")

    probs = predict_all(tf, model, rows, img_size, args.batch)
    preds = probs.argmax(axis=1)

    out_dir = out_root / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for c in classes:
        (out_dir / c).mkdir(parents=True, exist_ok=True)

    csv_rows = []
    n_correct = 0
    by_true = {}
    for r, pred_idx, p in zip(rows, preds, probs):
        pred_label = classes[int(pred_idx)]
        correct = pred_label == r["label"]
        n_correct += correct
        stat = by_true.setdefault(r["label"], {"n": 0, "correct": 0,
                                               "predicted_as": Counter()})
        stat["n"] += 1
        stat["correct"] += correct
        stat["predicted_as"][pred_label] += 1

        orig = Path(r["filename"]).name
        dst_name = f"ok__{orig}" if correct else f"true-{r['label']}__{orig}"
        shutil.copy2(r["path"], out_dir / pred_label / dst_name)

        csv_rows.append({
            "filename": f"{pred_label}/{dst_name}",
            "source_frame": r["filename"], "source_clip": r["clip"],
            "true_label": r["label"], "pred_label": pred_label,
            "confidence": round(float(p[int(pred_idx)]), 4),
            **{f"prob_{c}": round(float(p[i]), 4) for i, c in enumerate(classes)},
            "correct": int(correct),
        })

    fields = list(csv_rows[0].keys())
    with (out_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(csv_rows)

    acc = n_correct / len(rows)
    summary = {
        "dataset": name, "model": str(model_path), "classes": classes,
        "test_day": args.test_day, "test_frames": len(rows),
        "test_clips": len({r["clip"] for r in rows}),
        "accuracy": round(acc, 4),
        "per_true_label": {
            lab: {"n": s["n"], "correct": s["correct"],
                  "recall": round(s["correct"] / s["n"], 4),
                  "predicted_as": dict(s["predicted_as"])}
            for lab, s in sorted(by_true.items())},
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  accuracy={acc:.4f}  ({n_correct}/{len(rows)})")
    for lab, s in sorted(by_true.items()):
        print(f"    {lab:>10}: {s['correct']}/{s['n']} correct, "
              f"predicted_as={dict(s['predicted_as'])}")
    print(f"  saved -> {out_dir}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets-root", default=CONFIG["datasets_root"])
    ap.add_argument("--datasets", default=CONFIG["datasets"],
                    help="comma-separated dataset folder names; empty = all "
                         "with labels.csv AND a trained model")
    ap.add_argument("--models-root", default=CONFIG["models_root"],
                    help="root holding <dataset>/model.keras from the trainer")
    ap.add_argument("--out", default=CONFIG["out"])
    ap.add_argument("--test-day", type=int, default=CONFIG["test_day"],
                    help="collection day-of-month of the held-out test clips")
    ap.add_argument("--batch", type=int, default=CONFIG["batch"])
    in_notebook = any("ipykernel" in m or "colab" in m for m in sys.modules)
    args, _ = ap.parse_known_args([] if in_notebook else None)

    import tensorflow as tf  # noqa: PLC0415

    root = Path(args.datasets_root)
    models_root = Path(args.models_root)
    out_root = Path(args.out)
    if args.datasets.strip():
        dirs = [root / n.strip() for n in args.datasets.split(",") if n.strip()]
    else:
        dirs = sorted(p for p in root.iterdir() if (p / "labels.csv").is_file())
    pairs = []
    for d in dirs:
        if not (d / "labels.csv").is_file():
            print(f"[skip] no labels.csv: {d}")
            continue
        m = models_root / d.name
        if not (m / "model.keras").is_file():
            print(f"[skip] no trained model: {m / 'model.keras'}")
            continue
        pairs.append((d, m))
    if not pairs:
        raise SystemExit(f"Nothing to predict: no dataset under {root} has a "
                         f"model under {models_root}")
    print(f"Predicting {len(pairs)} dataset(s): {[d.name for d, _ in pairs]}")

    all_summaries = {}
    for d, m in pairs:
        s = predict_one(tf, d, m, out_root, args)
        if s:
            all_summaries[d.name] = s

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(
        json.dumps(all_summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{'='*70}\nALL DONE\n{'='*70}")
    for name, s in all_summaries.items():
        print(f"  {name}: accuracy={s['accuracy']:.4f} "
              f"({s['test_frames']} frames / {s['test_clips']} clips)")
    return 0


if __name__ == "__main__":
    rc = main()
    if not any("ipykernel" in m or "colab" in m for m in sys.modules):
        raise SystemExit(rc)
