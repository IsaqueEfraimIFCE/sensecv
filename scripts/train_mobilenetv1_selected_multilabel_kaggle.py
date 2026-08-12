# -*- coding: utf-8 -*-
"""Train ImageNet MobileNetV1 classifiers on the selected two-label dataset.

Paste this whole file into one Kaggle cell after running the universal
similarity selector. The selector's train_val/test membership is authoritative:
test is never resplit, and only train_val clips are divided into train and
validation.

Two independent, end-to-end MobileNetV1 models are trained:
  * obstacle_freepath: free_path / obstacle
  * deviation_lrn: left / right / none

Train and validation independently subsample only their background class at an
evenly spaced per-clip rate. Test keeps its natural distribution. The deviation
training set receives horizontal-flip copies; validation and test do not.
"""

import csv
import json
import random
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


CONFIG = {
    "dataset_dir": (
        "/kaggle/working/dinov2_similarity_selected_test_preserved"
    ),
    "output_dir": "/kaggle/working/mobilenetv1_selected_classifiers",
    "model_name": "mobilenetv1_100.ra4_e3600_r224_in1k",
    "image_size": 224,
    "batch_size": 64,
    "epochs": 40,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "val_fraction": 0.15,
    "seed": 43,
    "patience": 6,
    "num_workers": 2,
    "balance_train_validation_background": True,
    "deviation_flip_augmentation": True,
}


TASKS = {
    "obstacle_freepath": {
        "label_column": "obstacle_label",
        "classes": ["free_path", "obstacle"],
        "flip": False,
        "background": "free_path",
        "target_labels": ["obstacle"],
    },
    "deviation_lrn": {
        "label_column": "deviation_label",
        "classes": ["left", "right", "none"],
        "flip": True,
        "background": "none",
        "target_labels": ["left", "right"],
    },
}
REQUIRED_COLUMNS = {
    "filename",
    "split",
    "obstacle_label",
    "deviation_label",
    "source_clip",
}
NORMALIZATION = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_selected_rows(dataset_dir):
    labels_path = dataset_dir / "labels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(f"labels.csv not found: {labels_path}")
    rows = []
    with labels_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Selected labels.csv is missing: "
                + ", ".join(sorted(missing))
            )
        for source_row in reader:
            image_path = dataset_dir / source_row.get("filename", "")
            split = source_row.get("split", "").strip()
            clip = source_row.get("source_clip", "").strip()
            if (
                image_path.is_file()
                and split in {"train_val", "test"}
                and clip
            ):
                rows.append(
                    {
                        "path": str(image_path),
                        "clip": clip,
                        "split": split,
                        "obstacle_label": source_row[
                            "obstacle_label"
                        ].strip(),
                        "deviation_label": source_row[
                            "deviation_label"
                        ].strip(),
                    }
                )
    return rows


def split_train_val_by_clip(rows, label_column, val_fraction, seed):
    """Split whole clips using the same policy as the selected DINO trainer."""
    rng = np.random.default_rng(seed)
    rows_by_clip = defaultdict(list)
    for index, row in enumerate(rows):
        rows_by_clip[row["clip"]].append(index)
    clip_label = {
        clip: Counter(
            rows[index][label_column] for index in indices
        ).most_common(1)[0][0]
        for clip, indices in rows_by_clip.items()
    }
    clips_by_class = defaultdict(list)
    for clip, label in clip_label.items():
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

    train_indices = [
        index for index, row in enumerate(rows) if row["clip"] in train_clips
    ]
    val_indices = [
        index for index, row in enumerate(rows) if row["clip"] in val_clips
    ]
    if set(train_indices) & set(val_indices):
        raise AssertionError("A row was assigned to both train and validation.")
    if set(train_indices) | set(val_indices) != set(range(len(rows))):
        raise AssertionError(
            "The train/validation split dropped selected source rows."
        )
    return train_indices, val_indices, {
        "train_clips": len(train_clips),
        "val_clips": len(val_clips),
    }


def balance_background_by_clip(
    rows,
    label_column,
    background_label,
    target_labels,
    enabled,
):
    """Subsample background at evenly spaced per-clip intervals."""
    counts = Counter(row[label_column] for row in rows)
    if len(target_labels) == 1:
        requested_target = counts.get(target_labels[0], 0)
        target_rule = f"match_{target_labels[0]}"
    else:
        requested_target = (
            sum(counts.get(label, 0) for label in target_labels)
            + len(target_labels) // 2
        ) // len(target_labels)
        target_rule = "rounded_mean_" + "_".join(target_labels)
    available = counts.get(background_label, 0)
    target = min(available, requested_target)
    summary = {
        "enabled": bool(enabled),
        "applied": False,
        "label_column": label_column,
        "background_label": background_label,
        "target_labels": target_labels,
        "target_rule": target_rule,
        "source_frames": len(rows),
        "source_class_counts": dict(sorted(counts.items())),
        "source_background_frames": available,
        "requested_background_frames": requested_target,
        "selected_background_frames": available,
        "effective_frames": len(rows),
        "effective_class_counts": dict(sorted(counts.items())),
        "global_keep_fraction": 1.0,
        "clips": {},
    }
    if not enabled or available <= target:
        return list(rows), summary

    background_by_clip = defaultdict(list)
    for row in rows:
        if row[label_column] == background_label:
            background_by_clip[row["clip"]].append(row)

    allocations = {}
    remainders = []
    allocated = 0
    for clip, clip_rows in sorted(background_by_clip.items()):
        exact = len(clip_rows) * target / available
        quota = int(np.floor(exact))
        allocations[clip] = quota
        allocated += quota
        remainders.append((exact - quota, clip))
    for _, clip in sorted(
        remainders, key=lambda item: (-item[0], item[1])
    )[: target - allocated]:
        allocations[clip] += 1

    selected_ids = set()
    clip_summary = {}
    for clip, clip_rows in sorted(background_by_clip.items()):
        source_count = len(clip_rows)
        selected_count = allocations[clip]
        if selected_count:
            indices = np.floor(
                (np.arange(selected_count) + 0.5)
                * source_count
                / selected_count
            ).astype(int)
            selected_ids.update(id(clip_rows[index]) for index in indices)
        clip_summary[clip] = {
            "source_frames": source_count,
            "selected_frames": selected_count,
            "keep_fraction": round(selected_count / source_count, 6),
            "frame_stride": (
                None
                if selected_count == 0
                else round(source_count / selected_count, 6)
            ),
        }

    balanced = [
        row
        for row in rows
        if row[label_column] != background_label or id(row) in selected_ids
    ]
    effective_counts = Counter(row[label_column] for row in balanced)
    summary.update(
        {
            "applied": True,
            "selected_background_frames": target,
            "effective_frames": len(balanced),
            "effective_class_counts": dict(sorted(effective_counts.items())),
            "global_keep_fraction": round(target / available, 6),
            "clips": clip_summary,
        }
    )
    return balanced, summary


class SelectedImages(Dataset):
    def __init__(
        self,
        rows,
        label_column,
        label_to_index,
        transform,
        include_flipped=False,
    ):
        self.rows = rows
        self.label_column = label_column
        self.label_to_index = label_to_index
        self.transform = transform
        self.include_flipped = include_flipped
        self.flip_map = {"left": "right", "right": "left", "none": "none"}

    def __len__(self):
        multiplier = 2 if self.include_flipped else 1
        return len(self.rows) * multiplier

    def __getitem__(self, index):
        flipped = self.include_flipped and index >= len(self.rows)
        row = self.rows[index % len(self.rows)]
        with Image.open(row["path"]) as image:
            image = image.convert("RGB")
            if flipped:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            pixels = self.transform(image)
        label = row[self.label_column]
        if flipped:
            label = self.flip_map[label]
        return pixels, self.label_to_index[label]


def make_loader(
    rows,
    label_column,
    label_to_index,
    transform,
    device,
    shuffle,
    seed,
    include_flipped=False,
):
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        SelectedImages(
            rows,
            label_column,
            label_to_index,
            transform,
            include_flipped=include_flipped,
        ),
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

    recalls = []
    f1_scores = []
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


def create_imagenet_model(num_classes, device):
    try:
        model = timm.create_model(
            CONFIG["model_name"],
            pretrained=True,
            num_classes=num_classes,
        )
    except Exception as error:
        raise RuntimeError(
            "Could not load the ImageNet MobileNetV1 weights. Enable Internet "
            "in the Kaggle notebook and run again."
        ) from error
    return model.to(device)


def train_model(
    model,
    train_loader,
    val_loader,
    classes,
    device,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-7
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_loss = float("inf")
    best_state = None
    bad_epochs = 0
    epochs_completed = 0
    history = []

    for epoch in range(CONFIG["epochs"]):
        model.train()
        loss_sum = 0.0
        correct = 0
        examples = 0
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
            loss_sum += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            examples += batch_size

        validation = evaluate(model, val_loader, classes, device)
        train_loss = loss_sum / max(1, examples)
        train_accuracy = correct / max(1, examples)
        scheduler.step(validation["loss"])
        epochs_completed = epoch + 1
        history.append(
            {
                "epoch": epochs_completed,
                "train_loss": round(train_loss, 6),
                "train_accuracy": round(train_accuracy, 6),
                "validation_loss": validation["loss"],
                "validation_accuracy": validation["accuracy"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"  epoch {epochs_completed:02d}: "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_accuracy:.4f} "
            f"val_loss={validation['loss']:.4f} "
            f"val_acc={validation['accuracy']:.4f}"
        )
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= CONFIG["patience"]:
                break

    if best_state is None:
        raise RuntimeError("MobileNetV1 training produced no checkpoint.")
    model.load_state_dict(best_state)
    return best_state, history, epochs_completed


def class_counts(rows, label_column, classes):
    counts = Counter(row[label_column] for row in rows)
    return {label: counts.get(label, 0) for label in classes}


def effective_training_counts(rows, label_column, classes, flip_applied):
    counts = Counter(row[label_column] for row in rows)
    if flip_applied:
        counts["left"] += sum(
            row[label_column] == "right" for row in rows
        )
        counts["right"] += sum(
            row[label_column] == "left" for row in rows
        )
        counts["none"] += sum(
            row[label_column] == "none" for row in rows
        )
    return {label: counts.get(label, 0) for label in classes}


set_seed(CONFIG["seed"])
dataset_dir = Path(CONFIG["dataset_dir"])
output_dir = Path(CONFIG["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)
rows = read_selected_rows(dataset_dir)
trainval_rows = [row for row in rows if row["split"] == "train_val"]
test_rows = [row for row in rows if row["split"] == "test"]
if not trainval_rows or not test_rows:
    raise RuntimeError(
        "Selected train_val or test rows are empty. The selector must process "
        "frames from both source splits."
    )
clip_overlap = (
    {row["clip"] for row in trainval_rows}
    & {row["clip"] for row in test_rows}
)
if clip_overlap:
    raise RuntimeError(
        "Selected source has clips spanning train_val and test: "
        + ", ".join(sorted(clip_overlap)[:10])
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose(
    [
        transforms.Resize(
            (CONFIG["image_size"], CONFIG["image_size"]),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=NORMALIZATION["mean"],
            std=NORMALIZATION["std"],
        ),
    ]
)
results = {}

for task_index, (task_name, task) in enumerate(TASKS.items()):
    set_seed(CONFIG["seed"] + task_index)
    label_column = task["label_column"]
    classes = task["classes"]
    label_to_index = {
        label: index for index, label in enumerate(classes)
    }
    unknown = sorted({row[label_column] for row in rows} - set(classes))
    if unknown:
        raise ValueError(f"{task_name}: unknown labels {unknown}")

    train_local, val_local, clip_counts = split_train_val_by_clip(
        trainval_rows,
        label_column,
        CONFIG["val_fraction"],
        CONFIG["seed"],
    )
    train_before = [trainval_rows[index] for index in train_local]
    validation_before = [trainval_rows[index] for index in val_local]
    if not train_before or not validation_before:
        raise RuntimeError(f"{task_name}: train or validation is empty.")

    train_rows, training_balance = balance_background_by_clip(
        train_before,
        label_column,
        task["background"],
        task["target_labels"],
        CONFIG["balance_train_validation_background"],
    )
    validation_rows, validation_balance = balance_background_by_clip(
        validation_before,
        label_column,
        task["background"],
        task["target_labels"],
        CONFIG["balance_train_validation_background"],
    )
    flip_applied = bool(
        task["flip"] and CONFIG["deviation_flip_augmentation"]
    )

    train_loader = make_loader(
        train_rows,
        label_column,
        label_to_index,
        transform,
        device,
        True,
        CONFIG["seed"] + task_index,
        include_flipped=flip_applied,
    )
    validation_loader = make_loader(
        validation_rows,
        label_column,
        label_to_index,
        transform,
        device,
        False,
        CONFIG["seed"],
    )
    test_loader = make_loader(
        test_rows,
        label_column,
        label_to_index,
        transform,
        device,
        False,
        CONFIG["seed"],
    )

    print(f"\nStarting {task_name} from ImageNet MobileNetV1 weights")
    model = create_imagenet_model(len(classes), device)
    best_state, history, epochs_completed = train_model(
        model,
        train_loader,
        validation_loader,
        classes,
        device,
    )
    validation_metrics = evaluate(
        model, validation_loader, classes, device
    )
    test_metrics = evaluate(model, test_loader, classes, device)

    split_counts = {
        "train_before_balance": class_counts(
            train_before, label_column, classes
        ),
        "train_source": class_counts(train_rows, label_column, classes),
        "train_effective": effective_training_counts(
            train_rows, label_column, classes, flip_applied
        ),
        "validation_before_balance": class_counts(
            validation_before, label_column, classes
        ),
        "validation": class_counts(
            validation_rows, label_column, classes
        ),
        "test": class_counts(test_rows, label_column, classes),
    }
    checkpoint = {
        "model_state_dict": best_state,
        "model_name": CONFIG["model_name"],
        "pretrained_initialization": "imagenet",
        "classes": classes,
        "label_column": label_column,
        "image_size": CONFIG["image_size"],
        "normalization": NORMALIZATION,
        "source_dataset": str(dataset_dir),
        "selection": "universal_dinov2_cosine_selected_images_only",
        "training_flip_augmentation": flip_applied,
        "training_background_balance": training_balance,
        "validation_background_balance": validation_balance,
        "test_balancing_applied": False,
    }
    torch.save(checkpoint, output_dir / f"{task_name}.pt")

    results[task_name] = {
        "label_column": label_column,
        "classes": classes,
        "frames": {
            "train_before_balance": len(train_before),
            "train_source": len(train_rows),
            "train_effective": len(train_loader.dataset),
            "validation_before_balance": len(validation_before),
            "validation": len(validation_rows),
            "test": len(test_rows),
        },
        "clips": {
            **clip_counts,
            "test_clips": len({row["clip"] for row in test_rows}),
        },
        "class_counts": split_counts,
        "training_flip_augmentation": flip_applied,
        "training_background_balance": training_balance,
        "validation_background_balance": validation_balance,
        "test_balancing_applied": False,
        "epochs_completed": epochs_completed,
        "history": history,
        "validation": validation_metrics,
        "test": test_metrics,
    }
    print(
        f"{task_name}: train {len(train_loader.dataset)}, "
        f"val {len(validation_rows)}, test {len(test_rows)} | "
        f"val_f1={validation_metrics['macro_f1']:.4f} "
        f"test_f1={test_metrics['macro_f1']:.4f}"
    )
    print(
        f"  training background balance: {task['background']} "
        f"{training_balance['source_background_frames']} -> "
        f"{training_balance['selected_background_frames']} | "
        f"keep_fraction={training_balance['global_keep_fraction']:.6f}"
    )
    print(
        f"  validation background balance: {task['background']} "
        f"{validation_balance['source_background_frames']} -> "
        f"{validation_balance['selected_background_frames']} | "
        f"keep_fraction={validation_balance['global_keep_fraction']:.6f}"
    )
    for split_name in (
        "train_before_balance",
        "train_source",
        "train_effective",
        "validation_before_balance",
        "validation",
        "test",
    ):
        counts = split_counts[split_name]
        print(
            f"  {split_name}: "
            + ", ".join(
                f"{label}={counts[label]}" for label in classes
            )
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

metrics = {
    "source_dataset": str(dataset_dir),
    "selected_images": len(rows),
    "selected_train_val": len(trainval_rows),
    "selected_test": len(test_rows),
    "model_name": CONFIG["model_name"],
    "pretrained_initialization": "imagenet",
    "two_independent_models": True,
    "test_membership_preserved": True,
    "training_background_balancing_enabled": CONFIG[
        "balance_train_validation_background"
    ],
    "validation_background_balancing_enabled": CONFIG[
        "balance_train_validation_background"
    ],
    "test_balancing_applied": False,
    "class_weights": None,
    "val_fraction": CONFIG["val_fraction"],
    "seed": CONFIG["seed"],
    "tasks": results,
}
(output_dir / "metrics.json").write_text(
    json.dumps(metrics, indent=2), encoding="utf-8"
)
print(f"\nSaved both ImageNet MobileNetV1 classifiers to {output_dir}")
