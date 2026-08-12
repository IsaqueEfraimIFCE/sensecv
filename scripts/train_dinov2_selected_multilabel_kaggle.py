# -*- coding: utf-8 -*-
"""Train frozen-DINOv2 linear classifiers using only selected images.

Paste into one Kaggle cell after running the universal similarity selector.
The selector's train_val/test membership is authoritative: test is never
resplit, and only train_val clips are divided into train and validation.
"""

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms


CONFIG = {
    "dataset_dir": (
        "/kaggle/working/dinov2_similarity_selected_test_preserved"
    ),
    "output_dir": "/kaggle/working/dinov2_selected_classifiers",
    "dino_model": "dinov2_vits14",
    "image_size": 224,
    "batch_size": 64,
    "epochs": 40,
    "learning_rate": 1e-3,
    "val_fraction": 0.15,
    "seed": 43,
    "patience": 6,
    "num_workers": 2,
    "balance_train_validation_background": True,
    # Matches the existing deviation trainer: training rows receive flipped
    # copies, left/right swap, and none remains none.
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
    """Split whole clips without sampling, dropping, or balancing any row."""
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
    def __init__(self, rows, transform, horizontal_flip=False):
        self.rows = rows
        self.transform = transform
        self.horizontal_flip = horizontal_flip

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(self.rows[index]["path"]) as image:
            image = image.convert("RGB")
            if self.horizontal_flip:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            return self.transform(image)


def extract_embeddings(
    backbone, rows, transform, device, horizontal_flip=False
):
    loader = DataLoader(
        SelectedImages(rows, transform, horizontal_flip),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=device.type == "cuda",
    )
    embeddings = []
    backbone.eval()
    with torch.inference_mode():
        for images in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                features = backbone(images)
            if isinstance(features, dict):
                features = features["x_norm_clstoken"]
            embeddings.append(features.float().cpu())
    return torch.cat(embeddings)


def evaluate(classifier, features, labels, classes, device):
    classifier.eval()
    with torch.inference_mode():
        logits = classifier(features.to(device))
        predictions = logits.argmax(1).cpu()
        loss = F.cross_entropy(logits, labels.to(device)).item()
    matrix = torch.zeros((len(classes), len(classes)), dtype=torch.int64)
    for truth, prediction in zip(labels, predictions):
        matrix[truth, prediction] += 1
    recalls, f1_scores = [], []
    for index in range(len(classes)):
        tp = matrix[index, index].item()
        fp = matrix[:, index].sum().item() - tp
        fn = matrix[index, :].sum().item() - tp
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        recalls.append(recall)
        f1_scores.append(f1)
    return {
        "loss": round(loss, 6),
        "accuracy": round((predictions == labels).float().mean().item(), 6),
        "macro_f1": round(float(np.mean(f1_scores)), 6),
        "balanced_accuracy": round(float(np.mean(recalls)), 6),
        "confusion_matrix": matrix.tolist(),
    }


def train_linear_classifier(
    train_x,
    train_y,
    val_x,
    val_y,
    test_x,
    test_y,
    classes,
    device,
    seed,
):
    classifier = torch.nn.Linear(train_x.shape[1], len(classes)).to(device)
    optimizer = torch.optim.Adam(
        classifier.parameters(), lr=CONFIG["learning_rate"]
    )
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    best_loss = float("inf")
    best_state = None
    bad_epochs = 0
    epochs_completed = 0
    for epoch in range(CONFIG["epochs"]):
        classifier.train()
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(classifier(features), labels)
            loss.backward()
            optimizer.step()
        val_metrics = evaluate(
            classifier, val_x, val_y, classes, device
        )
        epochs_completed = epoch + 1
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in classifier.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= CONFIG["patience"]:
                break
    if best_state is None:
        raise RuntimeError("Classifier training produced no checkpoint.")
    classifier.load_state_dict(best_state)
    return (
        best_state,
        evaluate(classifier, val_x, val_y, classes, device),
        evaluate(classifier, test_x, test_y, classes, device),
        epochs_completed,
    )


seed = CONFIG["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

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
split_clip_overlap = (
    {row["clip"] for row in trainval_rows}
    & {row["clip"] for row in test_rows}
)
if split_clip_overlap:
    raise RuntimeError(
        "Selected source has clips spanning train_val and test: "
        + ", ".join(sorted(split_clip_overlap)[:10])
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
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ]
)
backbone = torch.hub.load(
    "facebookresearch/dinov2",
    CONFIG["dino_model"],
    pretrained=True,
).to(device)
for parameter in backbone.parameters():
    parameter.requires_grad = False

# Original embeddings are extracted exactly once for every selected image.
all_x = extract_embeddings(backbone, rows, transform, device)
row_index = {id(row): index for index, row in enumerate(rows)}
test_global = [row_index[id(row)] for row in test_rows]
test_x_shared = all_x[test_global]
results = {}

for task_index, (task_name, task) in enumerate(TASKS.items()):
    label_column = task["label_column"]
    classes = task["classes"]
    label_to_index = {
        label: index for index, label in enumerate(classes)
    }
    unknown = sorted(
        {row[label_column] for row in rows} - set(classes)
    )
    if unknown:
        raise ValueError(f"{task_name}: unknown labels {unknown}")

    train_local, val_local, clip_counts = split_train_val_by_clip(
        trainval_rows,
        label_column,
        CONFIG["val_fraction"],
        seed,
    )
    if not train_local or not val_local:
        raise RuntimeError(f"{task_name}: train or validation is empty.")
    train_rows_before_balance = [
        trainval_rows[index] for index in train_local
    ]
    val_rows_before_balance = [
        trainval_rows[index] for index in val_local
    ]
    train_rows, balance_summary = balance_background_by_clip(
        train_rows_before_balance,
        label_column,
        task["background"],
        task["target_labels"],
        CONFIG["balance_train_validation_background"],
    )
    val_rows, validation_balance = balance_background_by_clip(
        val_rows_before_balance,
        label_column,
        task["background"],
        task["target_labels"],
        CONFIG["balance_train_validation_background"],
    )
    train_global = [row_index[id(row)] for row in train_rows]
    val_global = [row_index[id(row)] for row in val_rows]
    train_x = all_x[train_global]
    val_x = all_x[val_global]
    test_x = test_x_shared
    train_y = torch.tensor(
        [label_to_index[row[label_column]] for row in train_rows],
        dtype=torch.long,
    )
    val_y = torch.tensor(
        [label_to_index[row[label_column]] for row in val_rows],
        dtype=torch.long,
    )
    test_y = torch.tensor(
        [label_to_index[row[label_column]] for row in test_rows],
        dtype=torch.long,
    )

    flip_applied = bool(
        task["flip"] and CONFIG["deviation_flip_augmentation"]
    )
    source_train_count = len(train_rows)
    if flip_applied:
        flipped_x = extract_embeddings(
            backbone,
            train_rows,
            transform,
            device,
            horizontal_flip=True,
        )
        flip_map = {"left": "right", "right": "left", "none": "none"}
        flipped_y = torch.tensor(
            [
                label_to_index[flip_map[row[label_column]]]
                for row in train_rows
            ],
            dtype=torch.long,
        )
        train_x = torch.cat([train_x, flipped_x])
        train_y = torch.cat([train_y, flipped_y])

    best_state, validation, test, epochs_completed = (
        train_linear_classifier(
            train_x,
            train_y,
            val_x,
            val_y,
            test_x,
            test_y,
            classes,
            device,
            seed + task_index,
        )
    )
    checkpoint = {
        "classifier_state_dict": best_state,
        "dino_model": CONFIG["dino_model"],
        "embedding_dim": all_x.shape[1],
        "classes": classes,
        "label_column": label_column,
        "image_size": CONFIG["image_size"],
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "source_dataset": str(dataset_dir),
        "selection": "universal_dinov2_cosine_selected_images_only",
        "training_flip_augmentation": flip_applied,
        "training_background_balance": balance_summary,
        "validation_background_balance": validation_balance,
        "balancing_applied": bool(
            balance_summary["applied"] or validation_balance["applied"]
        ),
        "class_weights": None,
        "foreground_rows_preserved": True,
        "validation_balancing_applied": validation_balance["applied"],
        "test_balancing_applied": False,
    }
    torch.save(checkpoint, output_dir / f"{task_name}.pt")
    split_counts = {}
    for split_name, selected, column in (
        (
            "train_before_balance",
            train_rows_before_balance,
            label_column,
        ),
        ("train_source", train_rows, label_column),
        (
            "validation_before_balance",
            val_rows_before_balance,
            label_column,
        ),
        ("validation", val_rows, label_column),
        ("test", test_rows, label_column),
    ):
        counts = Counter(row[column] for row in selected)
        split_counts[split_name] = {
            label: counts.get(label, 0) for label in classes
        }
    effective_counts = Counter(train_y.tolist())
    split_counts["train_effective"] = {
        label: effective_counts.get(label_to_index[label], 0)
        for label in classes
    }
    results[task_name] = {
        "label_column": label_column,
        "classes": classes,
        "frames": {
            "train_before_balance": len(train_rows_before_balance),
            "train_source": source_train_count,
            "train_effective": len(train_y),
            "validation_before_balance": len(val_rows_before_balance),
            "validation": len(val_rows),
            "test": len(test_rows),
        },
        "clips": {
            **clip_counts,
            "test_clips": len({row["clip"] for row in test_rows}),
        },
        "class_counts": split_counts,
        "training_flip_augmentation": flip_applied,
        "training_background_balance": balance_summary,
        "validation_background_balance": validation_balance,
        "balancing_applied": bool(
            balance_summary["applied"] or validation_balance["applied"]
        ),
        "class_weights": None,
        "foreground_rows_preserved": True,
        "validation_balancing_applied": validation_balance["applied"],
        "test_balancing_applied": False,
        "epochs_completed": epochs_completed,
        "validation": validation,
        "test": test,
    }
    print(
        f"{task_name}: train {len(train_y)}, val {len(val_rows)}, "
        f"test {len(test_rows)} | val_f1={validation['macro_f1']:.4f} "
        f"test_f1={test['macro_f1']:.4f}"
    )
    print(
        f"  training background balance: {task['background']} "
        f"{balance_summary['source_background_frames']} -> "
        f"{balance_summary['selected_background_frames']} | "
        f"keep_fraction={balance_summary['global_keep_fraction']:.6f}"
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

del backbone
if device.type == "cuda":
    torch.cuda.empty_cache()
metrics = {
    "source_dataset": str(dataset_dir),
    "selected_images": len(rows),
    "selected_train_val": len(trainval_rows),
    "selected_test": len(test_rows),
    "test_membership_preserved": True,
    "unselected_images_loaded": False,
    "training_background_balancing_enabled": CONFIG[
        "balance_train_validation_background"
    ],
    "validation_background_balancing_enabled": CONFIG[
        "balance_train_validation_background"
    ],
    "test_balancing_applied": False,
    "class_weights": None,
    "split_source_rows_preserved": True,
    "foreground_rows_preserved": True,
    "val_fraction": CONFIG["val_fraction"],
    "seed": seed,
    "tasks": results,
}
(output_dir / "metrics.json").write_text(
    json.dumps(metrics, indent=2), encoding="utf-8"
)
print(f"Saved both selected-image classifiers to {output_dir}")
