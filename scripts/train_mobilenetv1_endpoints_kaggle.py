# -*- coding: utf-8 -*-
"""Kaggle single-cell MobileNetV1 endpoint classifier comparison.

Paste this whole file into one Kaggle cell, edit CONFIG, and run it.

The input is one existing SenseCV all-frames dataset directory:
``.../kaggle_allframes_datasets_224/obstacle_freepath`` or
``.../kaggle_allframes_datasets_224/deviation_lrn``.

The script keeps the first and last frame of each decoded sequence, creates the
same clip-grouped split as the DINOv2 endpoint trainer, then trains MobileNetV1
end-to-end from ImageNet weights and from random weights. Day 22 is the fixed
test set. Undated clear/sem_obstaculos clips are treated as day-22 clips.
"""

import csv
import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    import timm
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "timm"]
    )
    import timm


# Edit only this block in Kaggle.
CONFIG = {
    "dataset_dir": (
        "/kaggle/input/datasets/isaqueefraim/dataset8/"
        "kaggle_allframes_datasets_224/obstacle_freepath"
    ),
    "output_dir": "/kaggle/working/mobilenetv1_endpoints_comparison",
    "model_name": "mobilenetv1_100.ra4_e3600_r224_in1k",
    "image_size": 224,
    "batch_size": 64,
    "epochs": 50,
    "imagenet_learning_rate": 1e-4,
    "scratch_learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "val_fraction": 0.15,
    "test_day": 22,
    "seed": 43,
    "patience": 8,
    "num_workers": 2,
}


DATE_RE = re.compile(r"(\d{2})-(\d{2})-(\d{4})")
FRAME_RE = re.compile(r"__f(\d+)$", re.IGNORECASE)
CANONICAL_CLASS_ORDERS = (
    ["free_path", "obstacle"],
    ["left", "right", "none"],
)
EXPERIMENTS = (
    {
        "name": "mobilenetv1_imagenet",
        "pretrained": True,
        "learning_rate": CONFIG["imagenet_learning_rate"],
    },
    {
        "name": "mobilenetv1_scratch",
        "pretrained": False,
        "learning_rate": CONFIG["scratch_learning_rate"],
    },
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def collection_day(clip):
    match = DATE_RE.search(clip or "")
    return int(match.group(1)) if match else None


def sequence_name(filename):
    path = Path(filename)
    return str(path.with_name(FRAME_RE.sub("", path.stem)))


def frame_number(row):
    value = str(row.get("frame_index", "")).strip()
    if value:
        try:
            return int(float(value))
        except ValueError:
            pass
    match = FRAME_RE.search(Path(row["filename"]).stem)
    return int(match.group(1)) if match else 0


def read_endpoint_rows(dataset_dir):
    """Read labels and keep the first/last frame of every decoded sequence."""
    labels_path = dataset_dir / "labels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(f"labels.csv not found: {labels_path}")

    sequences = defaultdict(list)
    with labels_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"filename", "label", "source_clip"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "labels.csv is missing required columns: "
                + ", ".join(sorted(missing))
            )

        for source_row in reader:
            filename = source_row.get("filename", "")
            image_path = dataset_dir / filename
            label = source_row.get("label", "").strip()
            clip = source_row.get("source_clip", "").strip()

            # Dataset rule: every undated clear/sem_obstaculos clip is from
            # 22-06-2026 and must therefore remain in the fixed test split.
            clear_name = f"{filename} {clip}".lower()
            if (
                DATE_RE.search(clip) is None
                and (
                    "sem_obstaculos" in clear_name
                    or source_row.get("source", "").strip().lower() == "clear"
                )
            ):
                clip = f"SenseCV-22-06-2026-clear_{clip or image_path.stem}"

            if image_path.is_file() and label and clip:
                row = {
                    "path": str(image_path),
                    "label": label,
                    "clip": clip,
                    "sequence": sequence_name(filename),
                    "frame": frame_number(source_row),
                }
                sequences[row["sequence"]].append(row)

    endpoints = []
    for rows in sequences.values():
        rows.sort(key=lambda row: (row["frame"], row["path"]))
        endpoints.append(rows[0])
        if rows[-1]["path"] != rows[0]["path"]:
            endpoints.append(rows[-1])
    return endpoints


def class_order(rows):
    present = {row["label"] for row in rows}
    for order in CANONICAL_CLASS_ORDERS:
        if present == set(order):
            return order
    return sorted(present)


def split_by_clip(rows, val_fraction, test_day, seed):
    """Fixed-day test plus stratified, clip-grouped train/validation."""
    rng = np.random.default_rng(seed)
    rows_by_clip = defaultdict(list)
    for row in rows:
        rows_by_clip[row["clip"]].append(row)

    clip_label = {
        clip: Counter(row["label"] for row in clip_rows).most_common(1)[0][0]
        for clip, clip_rows in rows_by_clip.items()
    }
    test_clips = {
        clip for clip in rows_by_clip if collection_day(clip) == test_day
    }

    clips_by_class = defaultdict(list)
    for clip, label in clip_label.items():
        if clip not in test_clips:
            clips_by_class[label].append(clip)

    train_clips, val_clips = set(), set()
    for clips in clips_by_class.values():
        clips = list(clips)
        rng.shuffle(clips)
        count = len(clips)
        val_count = (
            max(1, int(round(count * val_fraction)))
            if count > 2
            else (1 if count > 1 else 0)
        )
        val_count = min(val_count, max(0, count - 1))
        val_clips.update(clips[:val_count])
        train_clips.update(clips[val_count:])

    def collect(selected):
        return [row for row in rows if row["clip"] in selected]

    return (
        collect(train_clips),
        collect(val_clips),
        collect(test_clips),
        {
            "train_clips": len(train_clips),
            "val_clips": len(val_clips),
            "test_clips": len(test_clips),
        },
    )


class EndpointImages(Dataset):
    def __init__(self, rows, label_to_index, transform):
        self.rows = rows
        self.label_to_index = label_to_index
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            pixels = self.transform(image.convert("RGB"))
        return pixels, self.label_to_index[row["label"]]


def make_loader(rows, label_to_index, transform, device, shuffle, seed):
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        EndpointImages(rows, label_to_index, transform),
        batch_size=CONFIG["batch_size"],
        shuffle=shuffle,
        num_workers=CONFIG["num_workers"],
        pin_memory=device.type == "cuda",
        generator=generator,
    )


def evaluate(model, loader, classes, device):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_labels = []
    all_predictions = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
            predictions = logits.argmax(dim=1)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            all_labels.append(labels.cpu())
            all_predictions.append(predictions.cpu())

    labels = torch.cat(all_labels)
    predictions = torch.cat(all_predictions)
    matrix = torch.zeros((len(classes), len(classes)), dtype=torch.int64)
    for truth, prediction in zip(labels.tolist(), predictions.tolist()):
        matrix[truth, prediction] += 1

    recalls, f1_scores = [], []
    for index in range(len(classes)):
        true_positive = matrix[index, index].item()
        false_positive = matrix[:, index].sum().item() - true_positive
        false_negative = matrix[index, :].sum().item() - true_positive
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        recalls.append(recall)
        f1_scores.append(f1)

    return {
        "loss": round(total_loss / max(1, total_examples), 6),
        "accuracy": round((predictions == labels).float().mean().item(), 6),
        "macro_f1": round(float(np.mean(f1_scores)), 6),
        "balanced_accuracy": round(float(np.mean(recalls)), 6),
        "confusion_matrix": matrix.tolist(),
    }


def create_model(pretrained, num_classes, device):
    try:
        model = timm.create_model(
            CONFIG["model_name"],
            pretrained=pretrained,
            num_classes=num_classes,
        )
    except Exception as error:
        if pretrained:
            raise RuntimeError(
                "Could not load the ImageNet MobileNetV1 weights. "
                "Enable Internet in the Kaggle notebook and run again."
            ) from error
        raise
    return model.to(device)


def train_experiment(
    experiment,
    train_rows,
    val_rows,
    test_rows,
    label_to_index,
    classes,
    transform,
    output_dir,
    device,
):
    name = experiment["name"]
    pretrained = experiment["pretrained"]
    learning_rate = experiment["learning_rate"]
    set_seed(CONFIG["seed"])

    train_loader = make_loader(
        train_rows, label_to_index, transform, device, True, CONFIG["seed"]
    )
    val_loader = make_loader(
        val_rows, label_to_index, transform, device, False, CONFIG["seed"]
    )
    test_loader = make_loader(
        test_rows, label_to_index, transform, device, False, CONFIG["seed"]
    )

    model = create_model(pretrained, len(classes), device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-7
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda"
    )

    best_val_loss = float("inf")
    best_state = None
    bad_epochs = 0
    epochs_completed = 0
    history = []
    print(f"\nStarting {name} | pretrained={pretrained} | lr={learning_rate:g}")

    for epoch in range(CONFIG["epochs"]):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_examples = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = labels.size(0)
            train_loss_sum += loss.item() * batch_size
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_examples += batch_size

        train_loss = train_loss_sum / max(1, train_examples)
        train_accuracy = train_correct / max(1, train_examples)
        val_metrics = evaluate(model, val_loader, classes, device)
        scheduler.step(val_metrics["loss"])
        epochs_completed = epoch + 1
        current_lr = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epochs_completed,
                "learning_rate": current_lr,
                "train_loss": round(train_loss, 6),
                "train_accuracy": round(train_accuracy, 6),
                "validation_loss": val_metrics["loss"],
                "validation_accuracy": val_metrics["accuracy"],
                "validation_macro_f1": val_metrics["macro_f1"],
            }
        )
        print(
            f"{name} | epoch {epochs_completed:02d} | "
            f"train_loss={train_loss:.4f} | train_acc={train_accuracy:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_f1={val_metrics['macro_f1']:.4f} | lr={current_lr:.2e}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= CONFIG["patience"]:
                print(f"Early stopping {name} at epoch {epochs_completed}.")
                break

    if best_state is None:
        raise RuntimeError(f"No valid checkpoint produced for {name}.")

    model.load_state_dict(best_state)
    validation = evaluate(model, val_loader, classes, device)
    test = evaluate(model, test_loader, classes, device)
    checkpoint_path = output_dir / f"{name}.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "model_name": CONFIG["model_name"],
            "pretrained_initialization": pretrained,
            "classes": classes,
            "image_size": CONFIG["image_size"],
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        checkpoint_path,
    )
    result = {
        "experiment": name,
        "model_name": CONFIG["model_name"],
        "pretrained_initialization": pretrained,
        "learning_rate": learning_rate,
        "epochs_completed": epochs_completed,
        "best_validation_loss": round(best_val_loss, 6),
        "validation": validation,
        "test": test,
        "checkpoint": str(checkpoint_path),
        "history": history,
    }

    del model, optimizer, scheduler, scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def print_comparison(results):
    headers = (
        "experiment",
        "val_acc",
        "val_f1",
        "test_acc",
        "test_f1",
        "test_bal_acc",
    )
    rows = [
        (
            result["experiment"],
            f"{result['validation']['accuracy']:.4f}",
            f"{result['validation']['macro_f1']:.4f}",
            f"{result['test']['accuracy']:.4f}",
            f"{result['test']['macro_f1']:.4f}",
            f"{result['test']['balanced_accuracy']:.4f}",
        )
        for result in results
    ]
    widths = [
        max(len(headers[index]), max(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print("\nFINAL COMPARISON")
    print(
        " | ".join(
            headers[index].ljust(widths[index])
            for index in range(len(headers))
        )
    )
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(
            " | ".join(
                row[index].ljust(widths[index])
                for index in range(len(row))
            )
        )


set_seed(CONFIG["seed"])
dataset_dir = Path(CONFIG["dataset_dir"])
output_dir = Path(CONFIG["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)

rows = read_endpoint_rows(dataset_dir)
classes = class_order(rows)
label_to_index = {label: index for index, label in enumerate(classes)}
train_rows, val_rows, test_rows, clip_counts = split_by_clip(
    rows,
    CONFIG["val_fraction"],
    CONFIG["test_day"],
    CONFIG["seed"],
)
if not rows or not train_rows or not val_rows or not test_rows:
    raise RuntimeError(
        "The endpoint dataset or one split is empty. "
        "Check dataset_dir, source_clip dates, and test_day."
    )

print(
    f"{dataset_dir.name}: {len(rows)} endpoint frames, {len(classes)} classes | "
    f"train {len(train_rows)}, val {len(val_rows)}, test {len(test_rows)}"
)
for split_name, split_rows in (
    ("train", train_rows),
    ("validation", val_rows),
    ("test", test_rows),
):
    counts = Counter(row["label"] for row in split_rows)
    print(
        f"  {split_name}: "
        + ", ".join(f"{label}={counts.get(label, 0)}" for label in classes)
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
transform = transforms.Compose(
    [
        transforms.Resize(
            (CONFIG["image_size"], CONFIG["image_size"]),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ]
)

results = [
    train_experiment(
        experiment=experiment,
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
        label_to_index=label_to_index,
        classes=classes,
        transform=transform,
        output_dir=output_dir,
        device=device,
    )
    for experiment in EXPERIMENTS
]
comparison = {
    "dataset": dataset_dir.name,
    "selection": "first_and_last_frame_per_sequence",
    "classes": classes,
    "test_day": CONFIG["test_day"],
    "frames": {
        "all": len(rows),
        "train": len(train_rows),
        "validation": len(val_rows),
        "test": len(test_rows),
    },
    "class_counts": {
        split_name: dict(Counter(row["label"] for row in split_rows))
        for split_name, split_rows in (
            ("train", train_rows),
            ("validation", val_rows),
            ("test", test_rows),
        )
    },
    "clips": clip_counts,
    "experiments": results,
}
metrics_path = output_dir / "metrics_comparison.json"
metrics_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

print_comparison(results)
print(f"\nSaved comparison metrics to: {metrics_path}")
print(f"Saved models to: {output_dir}")
