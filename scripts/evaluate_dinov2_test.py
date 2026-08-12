# -*- coding: utf-8 -*-
"""Evaluate promoted SenseCV DINOv2 classifiers on the fixed test day.

Reads each dataset's labels.csv, selects every frame whose source_clip date has
the requested day-of-month (22 by default), restores the promoted model.pt, and
writes metrics.json plus predictions.csv. Test frames are never used for model
selection or fitting.

Kaggle example:
  python evaluate_dinov2_test.py \
      --datasets-root /kaggle/working/datasets \
      --models-root /kaggle/working/models_dinov2 \
      --out /kaggle/working/dinov2_test_metrics
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np


CONFIG = {
    "datasets_root": "/kaggle/working/datasets",
    "models_root": "/kaggle/working/models_dinov2",
    "datasets": "obstacle_freepath_dino,deviation_lrn_dino",
    "out": "/kaggle/working/dinov2_test_metrics",
    "test_day": 22,
    "batch": 64,
}

DATE_RE = re.compile(r"(\d{2})-(\d{2})-(\d{4})")
HF_MODELS = {
    "vits14": "facebook/dinov2-small",
    "vitb14": "facebook/dinov2-base",
}


def day_of(value):
    match = DATE_RE.search(value or "")
    return int(match.group(1)) if match else None


def read_test_rows(dataset_dir, test_day):
    rows = []
    with (dataset_dir / "labels.csv").open(
            encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            source_clip = row.get("source_clip") or ""
            if day_of(source_clip) != test_day:
                continue
            image_path = dataset_dir / row["filename"]
            if image_path.is_file():
                rows.append({
                    "path": str(image_path),
                    "filename": row["filename"],
                    "label": row["label"],
                    "source_clip": source_clip,
                    "frame_index": row.get("frame_index", ""),
                    "time_s": row.get("time_s", ""),
                })
    return rows


def classification_report(y_true, y_pred, classes):
    y_true = np.asarray(y_true, np.int64)
    y_pred = np.asarray(y_pred, np.int64)
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for truth, prediction in zip(y_true, y_pred):
        matrix[truth, prediction] += 1
    per_class = {}
    for index, name in enumerate(classes):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(matrix[index, :].sum()),
        }
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(np.mean([item["f1"] for item in per_class.values()])),
        "balanced_accuracy": float(np.mean(
            [item["recall"] for item in per_class.values()])),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }


def load_backbone(torch, checkpoint):
    model_name = checkpoint["model_name"]
    source = checkpoint.get("pretrained_source", "torch_hub")
    if source == "transformers":
        from transformers import AutoModel

        class HFBackbone(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, images):
                return self.model(pixel_values=images).last_hidden_state[:, 0]

        return HFBackbone(AutoModel.from_pretrained(HF_MODELS[model_name]))
    return torch.hub.load(
        "facebookresearch/dinov2", checkpoint["hub_model"], pretrained=True)


def evaluate_dataset(dataset_dir, model_path, output_dir, args):
    try:
        import torch
        import torch.nn.functional as functional
        from PIL import Image
        from torchvision import transforms
    except ImportError as exc:
        raise SystemExit("Install torch, torchvision, and Pillow") from exc

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    classes = list(checkpoint["classes"])
    label_to_index = {name: index for index, name in enumerate(classes)}
    rows = read_test_rows(dataset_dir, args.test_day)
    unknown = sorted({row["label"] for row in rows} - set(classes))
    if unknown:
        raise RuntimeError(f"Test labels absent from checkpoint classes: {unknown}")
    if not rows:
        raise RuntimeError(f"No day-{args.test_day} test images in {dataset_dir}")

    mean = checkpoint["normalization"]["mean"]
    std = checkpoint["normalization"]["std"]
    transform = transforms.Compose([
        transforms.Resize((checkpoint["img_size"], checkpoint["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    class Frames(torch.utils.data.Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            with Image.open(rows[index]["path"]) as image:
                tensor = transform(image.convert("RGB"))
            return tensor, label_to_index[rows[index]["label"]], index

    backbone = load_backbone(torch, checkpoint)
    feature_dim = checkpoint["state_dict"]["classifier.weight"].shape[1]

    class Classifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.dropout = torch.nn.Dropout(checkpoint.get("dropout", 0.0))
            self.classifier = torch.nn.Linear(feature_dim, len(classes))

        def forward(self, images):
            features = self.backbone(images)
            if isinstance(features, dict):
                features = features.get(
                    "x_norm_clstoken", features.get("pooler_output"))
            return self.classifier(self.dropout(features))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Classifier().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    loader = torch.utils.data.DataLoader(
        Frames(), batch_size=args.batch, shuffle=False, num_workers=0,
        pin_memory=torch.cuda.is_available())

    truth, predicted, probabilities = [], [], []
    loss_sum = 0.0
    with torch.no_grad():
        for images, labels, _indices in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_sum += float(functional.cross_entropy(
                logits, labels, reduction="sum").item())
            probs = logits.softmax(dim=1)
            truth.extend(labels.cpu().tolist())
            predicted.extend(probs.argmax(1).cpu().tolist())
            probabilities.extend(probs.cpu().tolist())

    report = classification_report(truth, predicted, classes)
    report.update({
        "dataset": dataset_dir.name,
        "model": str(model_path),
        "model_name": checkpoint["model_name"],
        "pretrained_source": checkpoint.get("pretrained_source"),
        "test_day": args.test_day,
        "test_images": len(rows),
        "test_clips": len({row["source_clip"] for row in rows}),
        "true_counts": dict(Counter(row["label"] for row in rows)),
        "loss": loss_sum / len(rows),
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    fields = ["filename", "source_clip", "frame_index", "time_s",
              "true_label", "predicted_label", "correct", "confidence"]
    fields.extend(f"prob_{name}" for name in classes)
    with (output_dir / "predictions.csv").open(
            "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row, truth_index, prediction_index, probs in zip(
                rows, truth, predicted, probabilities):
            output = {key: row[key] for key in
                      ("filename", "source_clip", "frame_index", "time_s")}
            output.update({
                "true_label": classes[truth_index],
                "predicted_label": classes[prediction_index],
                "correct": truth_index == prediction_index,
                "confidence": max(probs),
            })
            output.update({f"prob_{name}": probs[index]
                           for index, name in enumerate(classes)})
            writer.writerow(output)

    print(f"\n{dataset_dir.name}: n={len(rows)} loss={report['loss']:.4f} "
          f"accuracy={report['accuracy']:.4f} macro_f1={report['macro_f1']:.4f} "
          f"balanced_accuracy={report['balanced_accuracy']:.4f}")
    print(f"confusion_matrix (rows=true, cols=pred; classes={classes}):")
    for matrix_row in report["confusion_matrix"]:
        print(" ", matrix_row)
    for name, metrics in report["per_class"].items():
        print(f"  {name:>10}: P={metrics['precision']:.4f} "
              f"R={metrics['recall']:.4f} F1={metrics['f1']:.4f} "
              f"n={metrics['support']}")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", default=CONFIG["datasets_root"])
    parser.add_argument("--models-root", default=CONFIG["models_root"])
    parser.add_argument("--datasets", default=CONFIG["datasets"])
    parser.add_argument("--out", default=CONFIG["out"])
    parser.add_argument("--test-day", type=int, default=CONFIG["test_day"])
    parser.add_argument("--batch", type=int, default=CONFIG["batch"])
    notebook = any("ipykernel" in name or "colab" in name for name in sys.modules)
    args, _unknown = parser.parse_known_args([] if notebook else None)
    if args.batch < 1:
        parser.error("--batch must be >= 1")
    return args


def main():
    args = parse_args()
    datasets_root = Path(args.datasets_root)
    models_root = Path(args.models_root)
    out_root = Path(args.out)
    names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    summary = {}
    for name in names:
        dataset_dir = datasets_root / name
        model_path = models_root / name / "model.pt"
        if not (dataset_dir / "labels.csv").is_file():
            print(f"[skip] missing dataset: {dataset_dir}")
            continue
        if not model_path.is_file():
            print(f"[skip] missing model: {model_path}")
            continue
        summary[name] = evaluate_dataset(
            dataset_dir, model_path, out_root / name, args)
    if not summary:
        raise SystemExit("No matching dataset/model pairs were found")
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved evaluation to {out_root}")
    return 0


if __name__ == "__main__":
    result = main()
    if not any("ipykernel" in name or "colab" in name for name in sys.modules):
        raise SystemExit(result)
