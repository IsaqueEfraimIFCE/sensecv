# -*- coding: utf-8 -*-
"""Kaggle single-cell DINOv2 all-frame classifier.

Paste this whole file into one Kaggle cell, edit CONFIG, and run it.

The input is one existing SenseCV all-frames dataset directory, for example
``.../kaggle_allframes_datasets_224/obstacle_freepath`` or
``.../kaggle_allframes_datasets_224/deviation_lrn``. Its ``labels.csv`` must
contain ``filename``, ``label``, and ``source_clip``.

Pipeline:
1. exclude clips whose manifest ``OBSTACULO`` is ``CAIXA`` plus another name;
2. keep every available frame from each remaining clip;
3. merge the reviewed 24-07 UFC export dataset;
4. build all-frame images from auto-trim cuts whose dates/clips are absent from
   those two datasets, excluding every known CAIXA clip;
5. discard collection day 9 entirely;
6. hold out collection days 22 and 24 as test; the remaining clips are split
   into clip-grouped, class-stratified train and validation sets;
7. augment only the training split with horizontal flips that swap left/right;
8. extract frozen DINOv2 CLS embeddings once;
9. train one dense classification layer and select it by validation loss.

The model directory also receives
``built_trainval_auto_trim/<task>/{labels.csv,dataset_info.json,class folders}``
so the newly decoded train/validation pool is auditable.
"""

import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms


# Edit only this block in Kaggle.
CONFIG = {
    "dataset_dir": (
        "/kaggle/input/datasets/isaqueefraim/dataset8/"
        "kaggle_allframes_datasets_224/obstacle_freepath"
    ),
    "output_dir": "/kaggle/working/dinov2_all_frames",
    "dino_model": "dinov2_vits14",
    "image_size": 224,
    "batch_size": 64,
    "epochs": 40,
    "learning_rate": 1e-3,
    "val_fraction": 0.15,
    "excluded_days": [9],
    "test_days": [22, 24],
    "seed": 43,
    "patience": 6,
    "num_workers": 2,
    "left_right_flip_augmentation": True,
    # "auto" finds the sibling folder locally or anywhere under /kaggle/input.
    # An explicit entry may be the package root or its task subfolder.
    "additional_dataset_roots": ["auto"],
    "require_july24_data": True,
    # Source of additional train/validation cuts. "auto" finds it locally or
    # below /kaggle/input.
    "auto_trim_root": "auto",
    "require_auto_trim_data": True,
}


DATE_RE = re.compile(r"(\d{2})-(\d{2})-(\d{4})")
CLIP_ID_RE = re.compile(r"_(\d+)$")
CANONICAL_CLASS_ORDERS = (
    ["free_path", "obstacle"],
    ["left", "right", "none"],
)
# Read from data/labels/*.xlsx. Plain "CAIXA" is intentionally not excluded:
# these are only the clips whose OBSTACULO cell contains CAIXA plus a qualifier.
COMPOUND_CAIXA_CLIP_IDS = {
    "02-06-2026": frozenset(range(9, 29)),
    "06-06-2026": frozenset(
        {10, 11, 12, 30, 31, 32, 50, 51, 52, 70, 71, 72}
    ),
    "10-06-2026": frozenset(
        {5, 6, 7, 20, 21, 22, 35, 36, 37, 50, 51, 52}
    ),
}
ADDITIONAL_DATASET_FOLDER = "kaggle_24-07-2026_ufc_exports_unique_224"
AUTO_TRIM_RELATIVE = Path("Todos_cortes_auto_trim") / "deviation"


def collection_day(clip):
    match = DATE_RE.search(clip or "")
    return int(match.group(1)) if match else None


def is_compound_caixa_clip(clip):
    date_match = DATE_RE.search(clip or "")
    id_match = CLIP_ID_RE.search(clip or "")
    if date_match is None or id_match is None:
        return False
    date = "-".join(date_match.groups())
    return int(id_match.group(1)) in COMPOUND_CAIXA_CLIP_IDS.get(date, ())


def exclude_collection_days(rows, excluded_days):
    excluded_days = {int(day) for day in excluded_days}
    excluded_rows = [
        row for row in rows if collection_day(row["clip"]) in excluded_days
    ]
    kept_rows = [
        row for row in rows if collection_day(row["clip"]) not in excluded_days
    ]
    return kept_rows, {
        "days": sorted(excluded_days),
        "frames": len(excluded_rows),
        "clips": len({row["clip"] for row in excluded_rows}),
        "source_clips": sorted({row["clip"] for row in excluded_rows}),
    }


def read_training_rows(dataset_dir):
    labels_path = dataset_dir / "labels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(f"labels.csv not found: {labels_path}")

    rows = []
    excluded_clips = set()
    excluded_source_frames = 0
    with labels_path.open(encoding="utf-8-sig", newline="") as file:
        for source_row in csv.DictReader(file):
            image_path = dataset_dir / source_row.get("filename", "")
            label = source_row.get("label", "").strip()
            clip = source_row.get("source_clip", "").strip()
            if is_compound_caixa_clip(clip):
                excluded_clips.add(clip)
                excluded_source_frames += 1
                continue
            if image_path.is_file() and label and clip:
                rows.append(
                    {
                        "path": str(image_path),
                        "label": label,
                        "clip": clip,
                    }
                )

    return rows, {
        "clips": len(excluded_clips),
        "source_frames": excluded_source_frames,
        "source_clips": sorted(excluded_clips),
    }


def class_order(rows):
    present = {row["label"] for row in rows}
    for order in CANONICAL_CLASS_ORDERS:
        if present == set(order):
            return order
    return sorted(present)


def split_by_clip(rows, val_fraction, test_days, seed):
    """Use fixed test days and clip-grouped, class-stratified train/val."""
    rng = np.random.default_rng(seed)
    if isinstance(test_days, (int, float, str)):
        test_days = {int(test_days)}
    else:
        test_days = {int(day) for day in test_days}
    rows_by_clip = defaultdict(list)
    for row in rows:
        rows_by_clip[row["clip"]].append(row)

    clip_label = {
        clip: Counter(row["label"] for row in clip_rows).most_common(1)[0][0]
        for clip, clip_rows in rows_by_clip.items()
    }
    test_clips = {
        clip for clip in rows_by_clip if collection_day(clip) in test_days
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


def find_additional_dataset_dirs(primary_dataset_dir, configured_roots):
    task_name = primary_dataset_dir.name
    package_roots = []
    explicit_dirs = []
    for value in configured_roots or []:
        if str(value).strip().lower() == "auto":
            local_candidates = [
                Path.cwd() / "data" / "derived" / ADDITIONAL_DATASET_FOLDER,
                primary_dataset_dir.parent / ADDITIONAL_DATASET_FOLDER,
            ]
            try:
                local_candidates.append(
                    Path(__file__).resolve().parents[1]
                    / "data"
                    / "derived"
                    / ADDITIONAL_DATASET_FOLDER
                )
            except NameError:
                pass
            package_roots.extend(local_candidates)
            kaggle_input = Path("/kaggle/input")
            if kaggle_input.is_dir():
                package_roots.extend(
                    path
                    for path in kaggle_input.rglob(ADDITIONAL_DATASET_FOLDER)
                    if path.is_dir()
                )
                for marker in kaggle_input.rglob("build_info.json"):
                    try:
                        info = json.loads(marker.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    if (
                        info.get("collection") == "SenseCV-24-07-2026-UFC"
                        and info.get("exports") == 27
                    ):
                        package_roots.append(marker.parent)
            continue

        candidate = Path(value)
        if candidate.name == task_name and (candidate / "labels.csv").is_file():
            explicit_dirs.append(candidate)
        else:
            package_roots.append(candidate)

    candidates = explicit_dirs + [root / task_name for root in package_roots]
    found = []
    seen = set()
    for candidate in candidates:
        if not (candidate / "labels.csv").is_file():
            continue
        resolved = candidate.resolve()
        if resolved == primary_dataset_dir.resolve() or resolved in seen:
            continue
        seen.add(resolved)
        found.append(resolved)
    return found


def load_additional_rows(dataset_dirs, base_rows, classes):
    base_clips = {row["clip"] for row in base_rows}
    accepted_clips = set()
    accepted_rows = []
    skipped_overlap_clips = set()
    sources = []
    for dataset_dir in dataset_dirs:
        rows, exclusions = read_training_rows(dataset_dir)
        unknown_labels = sorted({row["label"] for row in rows} - set(classes))
        if unknown_labels:
            raise ValueError(
                f"{dataset_dir}: labels not present in primary dataset: "
                f"{unknown_labels}"
            )
        rows_by_clip = defaultdict(list)
        for row in rows:
            rows_by_clip[row["clip"]].append(row)
        source_accepted = 0
        for clip, clip_rows in rows_by_clip.items():
            if clip in base_clips or clip in accepted_clips:
                skipped_overlap_clips.add(clip)
                continue
            accepted_clips.add(clip)
            accepted_rows.extend(clip_rows)
            source_accepted += len(clip_rows)
        sources.append(
            {
                "dataset_dir": str(dataset_dir),
                "available_frames": len(rows),
                "accepted_frames": source_accepted,
                "excluded_compound_caixa": exclusions,
            }
        )
    return accepted_rows, {
        "dataset_dirs": [str(path) for path in dataset_dirs],
        "frames": len(accepted_rows),
        "clips": len(accepted_clips),
        "skipped_overlap_clips": sorted(skipped_overlap_clips),
        "sources": sources,
    }


def find_auto_trim_root(configured_root):
    if str(configured_root).strip().lower() != "auto":
        candidate = Path(configured_root)
        return candidate.resolve() if candidate.is_dir() else None

    candidates = [
        Path.cwd()
        / "data"
        / "derived"
        / "manifest_exports"
        / AUTO_TRIM_RELATIVE,
    ]
    try:
        candidates.append(
            Path(__file__).resolve().parents[1]
            / "data"
            / "derived"
            / "manifest_exports"
            / AUTO_TRIM_RELATIVE
        )
    except NameError:
        pass
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        candidates.extend(
            path
            for path in kaggle_input.rglob("deviation")
            if path.is_dir()
            and path.parent.name == "Todos_cortes_auto_trim"
        )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return None


def read_auto_trim_ledger(root):
    path = root / "sources.csv"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        return {
            row.get("output_folder", "").strip(): row
            for row in csv.DictReader(file)
            if row.get("output_folder", "").strip()
        }


def auto_trim_label(task_name, event_type, side):
    event = (event_type or "").strip().lower()
    direction = (side or "").strip().upper()
    if event == "desvio":
        if task_name == "obstacle_freepath":
            return "obstacle"
        return {"LEFT": "left", "RIGHT": "right"}.get(direction, "none")
    if event == "parada":
        return "obstacle" if task_name == "obstacle_freepath" else "none"
    if event == "livre":
        return "free_path" if task_name == "obstacle_freepath" else "none"
    return None


def build_auto_trim_rows(
    root,
    existing_rows,
    task_name,
    cache_dir,
    image_size,
    test_days,
    excluded_days,
):
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is required to build the auto-trim training data."
        ) from error

    existing_clips = {row["clip"] for row in existing_rows}
    existing_dates = {
        match.group(0)
        for row in existing_rows
        if (match := DATE_RE.search(row["clip"]))
    }
    test_days = {int(day) for day in test_days}
    excluded_days = {int(day) for day in excluded_days}
    ledger = read_auto_trim_ledger(root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    label_rows = []
    skip_counts = Counter()
    accepted_clips = set()
    source_folders = sorted(path for path in root.iterdir() if path.is_dir())

    for folder in source_folders:
        clip = folder.name
        date_match = DATE_RE.search(clip)
        if is_compound_caixa_clip(clip):
            skip_counts["caixa"] += 1
            continue
        if collection_day(clip) in excluded_days:
            skip_counts["excluded_day"] += 1
            continue
        if clip in existing_clips:
            skip_counts["existing_clip"] += 1
            continue
        if date_match and date_match.group(0) in existing_dates:
            skip_counts["existing_day"] += 1
            continue
        if collection_day(clip) in test_days:
            skip_counts["test_day"] += 1
            continue

        info = None
        info_path = folder / "cut_info.json"
        if info_path.is_file():
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
            except Exception:
                info = None
        if not info:
            info = ledger.get(clip)
        if not info:
            skip_counts["missing_label_metadata"] += 1
            continue
        event_type = info.get("event_type", "")
        side = info.get("side", "")
        label = auto_trim_label(task_name, event_type, side)
        if label is None:
            skip_counts["unknown_event_type"] += 1
            continue
        videos = sorted(folder.glob("*.mp4"))
        if not videos:
            skip_counts["missing_video"] += 1
            continue

        capture = cv2.VideoCapture(str(videos[0]))
        if not capture.isOpened():
            skip_counts["unreadable_video"] += 1
            continue
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_index = 0
        clip_frames = 0
        label_dir = cache_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if (
                image_size > 0
                and (
                    frame.shape[0] != image_size
                    or frame.shape[1] != image_size
                )
            ):
                frame = cv2.resize(
                    frame,
                    (image_size, image_size),
                    interpolation=cv2.INTER_AREA,
                )
            filename = f"auto_trim__{clip}__f{frame_index:05d}.jpg"
            image_path = label_dir / filename
            if not cv2.imwrite(
                str(image_path),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 95],
            ):
                capture.release()
                raise OSError(f"Failed to write {image_path}")
            time_s = frame_index / fps if fps > 0 else 0.0
            rows.append(
                {"path": str(image_path), "label": label, "clip": clip}
            )
            label_rows.append(
                {
                    "filename": f"{label}/{filename}",
                    "label": label,
                    "source": "auto_trim",
                    "event_type": event_type,
                    "side": side,
                    "source_clip": clip,
                    "frame_index": frame_index,
                    "time_s": round(time_s, 4),
                }
            )
            frame_index += 1
            clip_frames += 1
        capture.release()
        if clip_frames:
            accepted_clips.add(clip)
        else:
            skip_counts["no_frames"] += 1

    fields = [
        "filename",
        "label",
        "source",
        "event_type",
        "side",
        "source_clip",
        "frame_index",
        "time_s",
    ]
    with (cache_dir / "labels.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(label_rows)
    summary = {
        "source_root": str(root),
        "built_dataset_dir": str(cache_dir),
        "source_folders": len(source_folders),
        "accepted_clips": len(accepted_clips),
        "frames": len(rows),
        "skip_counts": dict(sorted(skip_counts.items())),
        "existing_dates": sorted(existing_dates),
        "test_days": sorted(test_days),
        "excluded_days": sorted(excluded_days),
    }
    (cache_dir / "dataset_info.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return rows, summary


def add_left_right_flip_augmentation(rows, classes, enabled):
    summary = {
        "enabled": bool(enabled),
        "applied": False,
        "label_map": {"left": "right", "right": "left", "none": "none"},
        "source_train_frames": len(rows),
        "flipped_train_frames": 0,
        "effective_train_frames": len(rows),
    }
    if not enabled or not {"left", "right"}.issubset(classes):
        return list(rows), summary

    label_map = summary["label_map"]
    flipped_rows = []
    for row in rows:
        flipped_row = dict(row)
        flipped_row["label"] = label_map.get(row["label"], row["label"])
        flipped_row["horizontal_flip"] = True
        flipped_rows.append(flipped_row)

    augmented_rows = list(rows) + flipped_rows
    summary.update(
        {
            "applied": True,
            "flipped_train_frames": len(flipped_rows),
            "effective_train_frames": len(augmented_rows),
        }
    )
    return augmented_rows, summary


class TrainingImages(Dataset):
    def __init__(self, rows, label_to_index, transform):
        self.rows = rows
        self.label_to_index = label_to_index
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            image = image.convert("RGB")
            if row.get("horizontal_flip"):
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            pixels = self.transform(image)
        return pixels, self.label_to_index[row["label"]]


def extract_embeddings(backbone, rows, label_to_index, transform, device):
    loader = DataLoader(
        TrainingImages(rows, label_to_index, transform),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=device.type == "cuda",
    )
    features, labels = [], []
    backbone.eval()
    with torch.inference_mode():
        for images, batch_labels in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                batch_features = backbone(images)
            if isinstance(batch_features, dict):
                batch_features = batch_features["x_norm_clstoken"]
            features.append(batch_features.float().cpu())
            labels.append(batch_labels)
    return torch.cat(features), torch.cat(labels)


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
        true_positive = matrix[index, index].item()
        false_positive = matrix[:, index].sum().item() - true_positive
        false_negative = matrix[index, :].sum().item() - true_positive
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
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


seed = CONFIG["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

dataset_dir = Path(CONFIG["dataset_dir"])
output_dir = Path(CONFIG["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)

rows, compound_caixa_exclusion = read_training_rows(dataset_dir)
classes = class_order(rows)
if not rows:
    raise RuntimeError(
        "The base all-frame dataset is empty. Check dataset_dir and labels.csv."
    )
additional_dataset_dirs = find_additional_dataset_dirs(
    dataset_dir, CONFIG["additional_dataset_roots"]
)
if CONFIG["require_july24_data"] and not additional_dataset_dirs:
    raise FileNotFoundError(
        f"Required supplemental package {ADDITIONAL_DATASET_FOLDER!r} was not "
        "found. Attach its ZIP to Kaggle or set "
        "CONFIG['additional_dataset_roots'] explicitly."
    )
additional_rows, additional_data = load_additional_rows(
    additional_dataset_dirs, rows, classes
)
if CONFIG["require_july24_data"] and not additional_rows:
    raise RuntimeError(
        "The July-24 dataset was found but supplied no new rows."
    )
base_frame_count = len(rows)
rows = list(rows) + additional_rows
rows, excluded_day_data = exclude_collection_days(
    rows, CONFIG["excluded_days"]
)
auto_trim_root = find_auto_trim_root(CONFIG["auto_trim_root"])
if CONFIG["require_auto_trim_data"] and auto_trim_root is None:
    raise FileNotFoundError(
        "Required Todos_cortes_auto_trim/deviation directory was not found. "
        "Attach it to Kaggle or set CONFIG['auto_trim_root'] explicitly."
    )
auto_trim_rows, auto_trim_build = (
    build_auto_trim_rows(
        auto_trim_root,
        rows,
        dataset_dir.name,
        output_dir / "built_trainval_auto_trim" / dataset_dir.name,
        CONFIG["image_size"],
        CONFIG["test_days"],
        CONFIG["excluded_days"],
    )
    if auto_trim_root is not None
    else (
        [],
        {
            "source_root": None,
            "accepted_clips": 0,
            "frames": 0,
            "skip_counts": {},
        },
    )
)
if CONFIG["require_auto_trim_data"] and not auto_trim_rows:
    raise RuntimeError(
        "The auto-trim source was found but supplied no new non-CAIXA rows "
        "after existing days and clips were excluded."
    )
rows.extend(auto_trim_rows)
classes = class_order(rows)
label_to_index = {label: index for index, label in enumerate(classes)}
source_train_rows, val_rows, test_rows, clip_counts = (
    split_by_clip(
        rows,
        CONFIG["val_fraction"],
        CONFIG["test_days"],
        seed,
    )
)
if not source_train_rows or not val_rows or not test_rows:
    raise RuntimeError(
        "The combined dataset or one split is empty. Check source_clip dates "
        "and CONFIG['test_days']."
    )
train_rows, flip_augmentation = add_left_right_flip_augmentation(
    source_train_rows,
    classes,
    CONFIG["left_right_flip_augmentation"],
)

print(
    f"{dataset_dir.name}: {len(rows)} frames, {len(classes)} classes | "
    f"train {len(train_rows)}, val {len(val_rows)}, test {len(test_rows)}"
)
print(
    "  excluded compound-CAIXA manifest clips: "
    f"{compound_caixa_exclusion['clips']} clips, "
    f"{compound_caixa_exclusion['source_frames']} source frames"
)
print(
    "  July-24 data merged into test: "
    f"{additional_data['frames']} frames from "
    f"{additional_data['clips']} clips"
)
print(
    "  excluded collection days "
    f"{excluded_day_data['days']}: {excluded_day_data['clips']} clips, "
    f"{excluded_day_data['frames']} frames"
)
print(
    "  auto-trim train/validation build: "
    f"{auto_trim_build['frames']} frames from "
    f"{auto_trim_build['accepted_clips']} new clips | "
    f"skipped {auto_trim_build['skip_counts']}"
)
print(
    "  left/right flip augmentation: "
    f"{'applied' if flip_augmentation['applied'] else 'not applicable'} | "
    f"{flip_augmentation['source_train_frames']} source -> "
    f"{flip_augmentation['effective_train_frames']} effective train frames"
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

train_x, train_y = extract_embeddings(
    backbone, train_rows, label_to_index, transform, device
)
val_x, val_y = extract_embeddings(
    backbone, val_rows, label_to_index, transform, device
)
test_x, test_y = extract_embeddings(
    backbone, test_rows, label_to_index, transform, device
)
del backbone
if device.type == "cuda":
    torch.cuda.empty_cache()

embedding_dim = train_x.shape[1]
classifier = torch.nn.Linear(embedding_dim, len(classes)).to(device)
optimizer = torch.optim.Adam(
    classifier.parameters(), lr=CONFIG["learning_rate"]
)
train_loader = DataLoader(
    TensorDataset(train_x, train_y),
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    generator=torch.Generator().manual_seed(seed),
)

best_val_loss = float("inf")
best_state = None
bad_epochs = 0
epochs_completed = 0

for epoch in range(CONFIG["epochs"]):
    classifier.train()
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(classifier(features), labels)
        loss.backward()
        optimizer.step()

    val_metrics = evaluate(classifier, val_x, val_y, classes, device)
    epochs_completed = epoch + 1
    if val_metrics["loss"] < best_val_loss:
        best_val_loss = val_metrics["loss"]
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in classifier.state_dict().items()
        }
        bad_epochs = 0
    else:
        bad_epochs += 1
        if bad_epochs >= CONFIG["patience"]:
            break

classifier.load_state_dict(best_state)
validation = evaluate(classifier, val_x, val_y, classes, device)
test = evaluate(classifier, test_x, test_y, classes, device)

checkpoint = {
    "classifier_state_dict": best_state,
    "dino_model": CONFIG["dino_model"],
    "embedding_dim": embedding_dim,
    "classes": classes,
    "image_size": CONFIG["image_size"],
    "normalization": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "training_augmentation": flip_augmentation,
    "additional_data": additional_data,
    "auto_trim_build": auto_trim_build,
    "excluded_day_data": excluded_day_data,
}
torch.save(checkpoint, output_dir / "model.pt")

metrics = {
    "dataset": dataset_dir.name,
    "selection": (
        "all_frames_no_balance_day_9_excluded_days_22_24_test"
    ),
    "excluded_compound_caixa": compound_caixa_exclusion,
    "additional_data": additional_data,
    "auto_trim_build": auto_trim_build,
    "excluded_day_data": excluded_day_data,
    "training_augmentation": flip_augmentation,
    "classes": classes,
    "excluded_days": CONFIG["excluded_days"],
    "test_days": CONFIG["test_days"],
    "frames": {
        "base": base_frame_count,
        "all": len(rows),
        "july24": len(additional_rows),
        "auto_trim_trainval_pool": len(auto_trim_rows),
        "train_source": len(source_train_rows),
        "train": len(train_rows),
        "validation": len(val_rows),
        "test": len(test_rows),
    },
    "clips": clip_counts,
    "epochs_completed": epochs_completed,
    "validation": validation,
    "test": test,
}
(output_dir / "metrics.json").write_text(
    json.dumps(metrics, indent=2), encoding="utf-8"
)

print(
    f"done | val_f1={validation['macro_f1']:.4f} "
    f"test_f1={test['macro_f1']:.4f} | {output_dir}"
)
