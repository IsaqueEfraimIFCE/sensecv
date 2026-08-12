# -*- coding: utf-8 -*-
"""Kaggle ConvLSTM trainer with a MobileNetV1 spatial backbone.

Paste this whole file into one Kaggle cell, edit CONFIG, and run it.

The data-source pipeline matches the supplied all-frame scripts:
1. read every base-dataset frame and exclude compound-CAIXA clips;
2. merge the reviewed 24-07 UFC package without duplicate clips;
3. decode new auto-trim clips whose dates/clips are not represented;
4. hold out collection days 22 and 24 as test;
5. split all other clips into class-stratified train/validation groups.

After the clip split, consecutive frames are converted into fixed-length
sequences. MobileNetV1 extracts a spatial feature map from every frame, a
ConvLSTM processes those feature maps over time, and the last hidden state is
pooled and classified. Background balancing and horizontal left/right
augmentation are performed at sequence level, so temporal continuity is kept.
"""

import csv
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

DATA_ONLY = os.environ.get("SENSECV_CONVLSTM_DATA_ONLY") == "1"

if not DATA_ONLY:
    try:
        import timm
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "timm"]
        )
        import timm


CONFIG = {
    "dataset_dir": (
        "/kaggle/input/datasets/isaqueefraim/dataset8/"
        "kaggle_allframes_datasets_224/obstacle_freepath"
    ),
    "lateral_dataset_dir": "auto",  # sibling deviation_lrn labels.csv
    "output_dir": "/kaggle/working/convlstm_mobilenetv1_all_frames",
    # First name is the current pretrained timm MobileNetV1 variant.
    # The code also tries the generic mobilenetv1_100 alias as a fallback.
    "backbone_name": "mobilenetv1_100.ra4_e3600_r224_in1k",
    "pretrained_backbone": True,
    "image_size": 224,
    "sequence_length": 8,
    "sequence_stride": 4,
    "pad_short_clips": True,
    "sequence_target": "last_frame",
    "batch_size": 8,
    "epochs": 40,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "feature_channels": 256,
    "convlstm_hidden_channels": 256,
    "convlstm_kernel_size": 3,
    "dropout": 0.30,
    "val_fraction": 0.20,
    "test_days": [22, 24],
    "seed": 43,
    "patience": 6,
    "num_workers": 2,
    "balance_background_classes": True,
    "left_right_flip_augmentation": True,
    "additional_dataset_roots": ["auto"],
    "require_july24_data": True,
    "auto_trim_root": "auto",
    "require_auto_trim_data": True,
    # Save an upload-ready, self-contained copy of the exact sequences used
    # by this run beneath output_dir. In Kaggle, use "Save Version" after the
    # run, then create a dataset from this output directory if desired.
    "save_dataset_used": True,
    "saved_dataset_dirname": "dataset_used",
    # A completed export is never replaced unless this is explicitly enabled.
    "overwrite_saved_dataset": False,
}


DATE_RE = re.compile(r"(\d{2})-(\d{2})-(\d{4})")
CLIP_ID_RE = re.compile(r"_(\d+)$")
CANONICAL_CLASS_ORDERS = (
    ["free_path", "obstacle"],
    ["left", "right", "none"],
)
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
NORMALIZATION = {
    "mean": [0.5, 0.5, 0.5],
    "std": [0.5, 0.5, 0.5],
}


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


def is_compound_caixa_clip(clip):
    date_match = DATE_RE.search(clip or "")
    id_match = CLIP_ID_RE.search(clip or "")
    if date_match is None or id_match is None:
        return False
    date = "-".join(date_match.groups())
    return int(id_match.group(1)) in COMPOUND_CAIXA_CLIP_IDS.get(date, ())


def resolve_lateral_dataset_dir(dataset_dir):
    configured = CONFIG["lateral_dataset_dir"]
    return dataset_dir.parent / "deviation_lrn" if str(configured).lower() == "auto" else Path(configured)


def read_lateral_labels(dataset_dir):
    labels_path = resolve_lateral_dataset_dir(dataset_dir) / "labels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(f"lateral labels.csv not found: {labels_path}")
    with labels_path.open(encoding="utf-8-sig", newline="") as file:
        labels = {(r.get("source_clip", "").strip(), r.get("frame_index", "").strip()): r.get("label", "").strip() for r in csv.DictReader(file)}
    return labels


def read_training_rows(dataset_dir):
    labels_path = dataset_dir / "labels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(f"labels.csv not found: {labels_path}")
    lateral_labels = read_lateral_labels(dataset_dir)
    rows = []
    excluded_clips = set()
    excluded_source_frames = 0
    with labels_path.open(encoding="utf-8-sig", newline="") as file:
        for source_row in csv.DictReader(file):
            image_path = dataset_dir / source_row.get("filename", "")
            label = source_row.get("label", "").strip()
            clip = source_row.get("source_clip", "").strip()
            lateral_label = lateral_labels.get((clip, source_row.get("frame_index", "").strip()))
            clear_name = f"{source_row.get('filename', '')} {clip}".lower()
            if (
                DATE_RE.search(clip) is None
                and (
                    "sem_obstaculos" in clear_name
                    or source_row.get("source", "").strip().lower() == "clear"
                )
            ):
                clip = f"SenseCV-22-06-2026-clear_{clip or image_path.stem}"
            if is_compound_caixa_clip(clip):
                excluded_clips.add(clip)
                excluded_source_frames += 1
                continue
            if image_path.is_file() and label and clip and lateral_label:
                rows.append(
                    {
                        "path": str(image_path),
                        "label": label,
                        "lateral_label": lateral_label,
                        "clip": clip,
                    }
                )
            elif image_path.is_file() and label and clip:
                raise ValueError(f"Missing paired lateral label for {clip!r}")
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


def balance_background_frames(rows, classes, enabled):
    """Evenly subsample background frames within clips to match foreground."""
    background = (
        "free_path"
        if "free_path" in classes
        else ("none" if "none" in classes else None)
    )
    summary = {
        "enabled": bool(enabled),
        "applied": False,
        "background_class": background,
        "foreground_classes": (
            [label for label in classes if label != background]
            if background
            else []
        ),
        "source_frames": len(rows),
        "source_background_frames": 0,
        "target_background_frames": 0,
        "selected_background_frames": 0,
        "effective_frames": len(rows),
        "clips": {},
    }
    if not enabled or background is None:
        return list(rows), summary

    background_by_clip = defaultdict(list)
    foreground_rows = []
    for row in rows:
        if row["label"] == background:
            background_by_clip[row["clip"]].append(row)
        else:
            foreground_rows.append(row)
    available = sum(len(values) for values in background_by_clip.values())
    target = min(available, len(foreground_rows))
    summary["source_background_frames"] = available
    summary["target_background_frames"] = target
    if available <= target:
        summary["selected_background_frames"] = available
        return list(rows), summary

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
        }
    balanced = [
        row
        for row in rows
        if row["label"] != background or id(row) in selected_ids
    ]
    summary.update(
        {
            "applied": True,
            "selected_background_frames": target,
            "effective_frames": len(balanced),
            "clips": clip_summary,
        }
    )
    return balanced, summary


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


def auto_trim_labels(event_type, side):
    event = (event_type or "").strip().lower()
    direction = (side or "").strip().upper()
    if event == "desvio":
        return "obstacle", {"LEFT": "left", "RIGHT": "right"}.get(direction, "none")
    if event == "parada":
        return "obstacle", "none"
    if event == "livre":
        return "free_path", "none"
    return None, None


def build_auto_trim_rows(
    root,
    existing_rows,
    task_name,
    cache_dir,
    image_size,
    test_days,
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
        label, lateral_label = auto_trim_labels(event_type, side)
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
                {"path": str(image_path), "label": label, "lateral_label": lateral_label, "clip": clip}
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
    }
    (cache_dir / "dataset_info.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return rows, summary

def build_sequences(rows, sequence_length, stride, pad_short_clips, target_mode):
    """Convert frame rows into ordered, non-crossing clip sequences."""
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1.")
    if stride < 1:
        raise ValueError("sequence_stride must be at least 1.")
    if target_mode not in {"last_frame", "majority"}:
        raise ValueError("sequence_target must be 'last_frame' or 'majority'.")

    rows_by_clip = defaultdict(list)
    for source_order, row in enumerate(rows):
        ordered_row = dict(row)
        ordered_row["_source_order"] = source_order
        rows_by_clip[row["clip"]].append(ordered_row)

    samples = []
    short_clips = 0
    padded_sequences = 0
    clips_without_sequences = 0

    for clip, clip_rows in rows_by_clip.items():
        # labels.csv and auto-trim decoding are already chronological. The
        # source-order key preserves that order without relying on filenames.
        clip_rows.sort(key=lambda row: row["_source_order"])
        frame_count = len(clip_rows)
        if frame_count < sequence_length:
            if not pad_short_clips or frame_count == 0:
                clips_without_sequences += 1
                continue
            padded = clip_rows + [clip_rows[-1]] * (sequence_length - frame_count)
            windows = [padded]
            short_clips += 1
            padded_sequences += 1
        else:
            starts = list(range(0, frame_count - sequence_length + 1, stride))
            final_start = frame_count - sequence_length
            if not starts or starts[-1] != final_start:
                starts.append(final_start)
            windows = [
                clip_rows[start : start + sequence_length] for start in starts
            ]

        for window in windows:
            if target_mode == "last_frame":
                label = window[-1]["label"]
            else:
                label = Counter(row["label"] for row in window).most_common(1)[0][0]
            samples.append(
                {
                    "frames": window,
                    "label": label,
                    "lateral_label": window[-1]["lateral_label"] if target_mode == "last_frame" else Counter(row["lateral_label"] for row in window).most_common(1)[0][0],
                    "clip": clip,
                    "horizontal_flip": False,
                }
            )

    summary = {
        "source_frames": len(rows),
        "source_clips": len(rows_by_clip),
        "sequences": len(samples),
        "sequence_length": sequence_length,
        "sequence_stride": stride,
        "target_mode": target_mode,
        "short_clips_padded": short_clips,
        "padded_sequences": padded_sequences,
        "clips_without_sequences": clips_without_sequences,
    }
    return samples, summary


def balance_background_sequences(samples, classes, enabled):
    """Evenly subsample background sequences across clips."""
    background = (
        "free_path"
        if "free_path" in classes
        else ("none" if "none" in classes else None)
    )
    summary = {
        "enabled": bool(enabled),
        "applied": False,
        "background_class": background,
        "foreground_classes": (
            [label for label in classes if label != background]
            if background
            else []
        ),
        "source_sequences": len(samples),
        "source_background_sequences": 0,
        "target_background_sequences": 0,
        "selected_background_sequences": 0,
        "effective_sequences": len(samples),
        "clips": {},
    }
    if not enabled or background is None:
        return list(samples), summary

    background_by_clip = defaultdict(list)
    foreground_samples = []
    for sample in samples:
        if sample["label"] == background:
            background_by_clip[sample["clip"]].append(sample)
        else:
            foreground_samples.append(sample)

    available = sum(len(values) for values in background_by_clip.values())
    target = min(available, len(foreground_samples))
    summary["source_background_sequences"] = available
    summary["target_background_sequences"] = target

    if available <= target:
        summary["selected_background_sequences"] = available
        return list(samples), summary

    allocations = {}
    remainders = []
    allocated = 0
    for clip, clip_samples in sorted(background_by_clip.items()):
        exact = len(clip_samples) * target / available
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
    for clip, clip_samples in sorted(background_by_clip.items()):
        source_count = len(clip_samples)
        selected_count = allocations[clip]
        if selected_count:
            indices = np.floor(
                (np.arange(selected_count) + 0.5)
                * source_count
                / selected_count
            ).astype(int)
            selected_ids.update(id(clip_samples[index]) for index in indices)
        clip_summary[clip] = {
            "source_sequences": source_count,
            "selected_sequences": selected_count,
            "keep_fraction": round(selected_count / source_count, 6),
        }

    balanced = [
        sample
        for sample in samples
        if sample["label"] != background or id(sample) in selected_ids
    ]
    summary.update(
        {
            "applied": True,
            "selected_background_sequences": target,
            "effective_sequences": len(balanced),
            "clips": clip_summary,
        }
    )
    return balanced, summary


def add_left_right_flip_augmentation(samples, classes, enabled):
    summary = {
        "enabled": bool(enabled),
        "applied": False,
        "label_map": {"left": "right", "right": "left", "none": "none"},
        "source_train_sequences": len(samples),
        "flipped_train_sequences": 0,
        "effective_train_sequences": len(samples),
    }
    if not enabled or not {"left", "right"}.issubset(classes):
        return list(samples), summary

    flipped_samples = []
    for sample in samples:
        flipped = dict(sample)
        flipped["lateral_label"] = summary["label_map"].get(
            sample["lateral_label"], sample["lateral_label"]
        )
        flipped["horizontal_flip"] = True
        flipped_samples.append(flipped)

    augmented = list(samples) + flipped_samples
    summary.update(
        {
            "applied": True,
            "flipped_train_sequences": len(flipped_samples),
            "effective_train_sequences": len(augmented),
        }
    )
    return augmented, summary


def save_dataset_used(output_dir, split_samples, classes):
    """Save the exact sequence inputs and transformations used by training.

    Images are copied once, while a row in each sequence CSV records every
    frame position. This preserves padded frames and augmented horizontal
    flips without duplicating image files.
    """
    dataset_dir = output_dir / CONFIG["saved_dataset_dirname"]
    if dataset_dir.exists():
        if not CONFIG["overwrite_saved_dataset"]:
            raise FileExistsError(
                f"Dataset export already exists: {dataset_dir}. Set "
                "overwrite_saved_dataset=True to replace it."
            )
        shutil.rmtree(dataset_dir)
    images_dir = dataset_dir / "images"
    sequences_dir = dataset_dir / "sequences"
    images_dir.mkdir(parents=True)
    sequences_dir.mkdir()

    copied_images = {}
    image_records = {}
    split_summary = {}
    sequence_fields = [
        "sequence_id",
        "frame_position",
        "image_filename",
        "label",
        "lateral_label",
        "clip",
        "horizontal_flip",
    ]
    for split_name, samples in split_samples.items():
        sequence_path = sequences_dir / f"{split_name}.csv"
        sequence_count = 0
        with sequence_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=sequence_fields)
            writer.writeheader()
            for sequence_id, sample in enumerate(samples):
                sequence_count += 1
                for frame_position, row in enumerate(sample["frames"]):
                    source_path = Path(row["path"])
                    source_key = str(source_path.resolve())
                    if source_key not in copied_images:
                        digest = hashlib.sha256(
                            source_key.encode("utf-8")
                        ).hexdigest()[:16]
                        filename = f"{digest}_{source_path.name}"
                        destination = images_dir / filename
                        shutil.copy2(source_path, destination)
                        copied_images[source_key] = filename
                        image_records[filename] = {
                            "image_filename": f"images/{filename}",
                            "label": row["label"],
                            "lateral_label": row["lateral_label"],
                            "clip": row["clip"],
                        }
                    writer.writerow(
                        {
                            "sequence_id": sequence_id,
                            "frame_position": frame_position,
                            "image_filename": "images/"
                            + copied_images[source_key],
                            "label": sample["label"],
                            "lateral_label": sample["lateral_label"],
                            "clip": sample["clip"],
                            "horizontal_flip": sample.get(
                                "horizontal_flip", False
                            ),
                        }
                    )
        split_summary[split_name] = {
            "sequences": sequence_count,
            "sequence_manifest": f"sequences/{split_name}.csv",
        }

    with (dataset_dir / "frames.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=["image_filename", "label", "lateral_label", "clip"]
        )
        writer.writeheader()
        writer.writerows(
            image_records[name] for name in sorted(image_records)
        )
    export_info = {
        "format": "sensecv_convlstm_sequence_dataset_v1",
        "classes": classes,
        "lateral_classes": ["left", "right", "none"],
        "images": len(copied_images),
        "splits": split_summary,
        "sequence_length": CONFIG["sequence_length"],
        "sequence_stride": CONFIG["sequence_stride"],
        "sequence_target": CONFIG["sequence_target"],
        "image_size": CONFIG["image_size"],
        "normalization": NORMALIZATION,
        "notes": (
            "Sequence manifests are the authoritative split definition. "
            "horizontal_flip=True means flip each listed image at load time."
        ),
    }
    (dataset_dir / "dataset_info.json").write_text(
        json.dumps(export_info, indent=2), encoding="utf-8"
    )
    return {"path": str(dataset_dir), **export_info}


class TrainingSequences(Dataset):
    def __init__(self, samples, label_to_index, lateral_to_index, transform):
        self.samples = samples
        self.label_to_index = label_to_index
        self.lateral_to_index = lateral_to_index
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        tensors = []
        for row in sample["frames"]:
            with Image.open(row["path"]) as image:
                image = image.convert("RGB")
                if sample.get("horizontal_flip"):
                    image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                tensors.append(self.transform(image))
        sequence = torch.stack(tensors, dim=0)
        return sequence, self.label_to_index[sample["label"]], self.lateral_to_index[sample["lateral_label"]]


def make_loader(samples, label_to_index, lateral_to_index, transform, device, shuffle, seed):
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        TrainingSequences(samples, label_to_index, lateral_to_index, transform),
        batch_size=CONFIG["batch_size"],
        shuffle=shuffle,
        num_workers=CONFIG["num_workers"],
        pin_memory=device.type == "cuda",
        generator=generator,
        persistent_workers=CONFIG["num_workers"] > 0,
    )


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("ConvLSTM kernel size must be odd.")
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(self, inputs, state=None):
        if state is None:
            batch, _, height, width = inputs.shape
            hidden = inputs.new_zeros(
                batch, self.hidden_channels, height, width
            )
            cell = inputs.new_zeros(
                batch, self.hidden_channels, height, width
            )
        else:
            hidden, cell = state

        combined = torch.cat([inputs, hidden], dim=1)
        input_gate, forget_gate, output_gate, candidate = self.gates(
            combined
        ).chunk(4, dim=1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        output_gate = torch.sigmoid(output_gate)
        candidate = torch.tanh(candidate)
        cell = forget_gate * cell + input_gate * candidate
        hidden = output_gate * torch.tanh(cell)
        return hidden, cell


class MobileNetV1ConvLSTM(nn.Module):
    def __init__(
        self,
        num_classes,
        num_lateral_classes,
        backbone_name,
        pretrained,
        projected_channels,
        hidden_channels,
        kernel_size,
        dropout,
    ):
        super().__init__()
        candidate_names = []
        requested_names = [backbone_name]
        if backbone_name and not backbone_name.startswith("hf_hub:"):
            requested_names.append(f"hf_hub:timm/{backbone_name}")
        requested_names.append("mobilenetv1_100")
        for name in requested_names:
            if name and name not in candidate_names:
                candidate_names.append(name)

        errors = []
        backbone = None
        selected_name = None
        for name in candidate_names:
            try:
                backbone = timm.create_model(
                    name,
                    pretrained=pretrained,
                    features_only=True,
                )
                selected_name = name
                break
            except Exception as error:
                errors.append(f"{name}: {error}")

        if backbone is None:
            raise RuntimeError(
                "Could not create a timm MobileNetV1 feature extractor. "
                "When pretrained_backbone=True, enable Internet in Kaggle. "
                "Attempts: " + " | ".join(errors)
            )

        self.backbone = backbone
        self.backbone_name = selected_name
        backbone_channels = self.backbone.feature_info.channels()[-1]
        self.feature_projection = nn.Sequential(
            nn.Conv2d(
                backbone_channels,
                projected_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(projected_channels),
            nn.SiLU(inplace=True),
        )
        self.temporal = ConvLSTMCell(
            input_channels=projected_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
        )
        self.dropout = nn.Dropout(dropout)
        self.obstacle_classifier = nn.Linear(hidden_channels, num_classes)
        self.lateral_classifier = nn.Linear(hidden_channels, num_lateral_classes)

    def forward(self, sequences):
        if sequences.ndim != 5:
            raise ValueError(
                "Expected input shape [batch, time, channels, height, width]."
            )
        batch, time, channels, height, width = sequences.shape
        frames = sequences.reshape(batch * time, channels, height, width)
        feature_maps = self.backbone(frames)[-1]
        feature_maps = self.feature_projection(feature_maps)
        _, projected_channels, feature_height, feature_width = feature_maps.shape
        feature_maps = feature_maps.reshape(
            batch,
            time,
            projected_channels,
            feature_height,
            feature_width,
        )

        state = None
        for time_index in range(time):
            state = self.temporal(feature_maps[:, time_index], state)
        hidden, _ = state
        pooled = F.adaptive_avg_pool2d(hidden, output_size=1).flatten(1)
        pooled = self.dropout(pooled)
        return self.obstacle_classifier(pooled), self.lateral_classifier(pooled)


def head_metrics(loss_sum, labels, predictions, classes):
    matrix = torch.zeros((len(classes), len(classes)), dtype=torch.int64)
    for truth, prediction in zip(labels.tolist(), predictions.tolist()):
        matrix[truth, prediction] += 1
    recalls, f1_scores = [], []
    per_class = {}
    for index, name in enumerate(classes):
        tp = matrix[index, index].item(); fp = matrix[:, index].sum().item() - tp; fn = matrix[index, :].sum().item() - tp
        precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        recalls.append(recall); f1_scores.append(f1)
        per_class[name] = {"precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6), "support": int(matrix[index, :].sum())}
    return {"loss": round(loss_sum / max(1, len(labels)), 6), "accuracy": round((predictions == labels).float().mean().item(), 6), "macro_f1": round(float(np.mean(f1_scores)), 6), "balanced_accuracy": round(float(np.mean(recalls)), 6), "confusion_matrix": matrix.tolist(), "per_class": per_class}


def evaluate(model, loader, classes, lateral_classes, device):
    model.eval(); obstacle_loss = lateral_loss = 0.0; obst_labels=[]; obst_preds=[]; lat_labels=[]; lat_preds=[]
    with torch.inference_mode():
        for sequences, labels, lateral_labels in loader:
            sequences=sequences.to(device, non_blocking=True); labels=labels.to(device, non_blocking=True); lateral_labels=lateral_labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                obstacle_logits, lateral_logits = model(sequences)
                lo=F.cross_entropy(obstacle_logits, labels); ll=F.cross_entropy(lateral_logits, lateral_labels)
            obstacle_loss += lo.item()*labels.size(0); lateral_loss += ll.item()*labels.size(0)
            obst_labels.append(labels.cpu()); obst_preds.append(obstacle_logits.argmax(1).cpu()); lat_labels.append(lateral_labels.cpu()); lat_preds.append(lateral_logits.argmax(1).cpu())
    obstacle=head_metrics(obstacle_loss, torch.cat(obst_labels), torch.cat(obst_preds), classes)
    lateral=head_metrics(lateral_loss, torch.cat(lat_labels), torch.cat(lat_preds), lateral_classes)
    return {"loss": round(obstacle["loss"] + lateral["loss"], 6), "obstacle": obstacle, "lateral": lateral}

def create_model(num_classes, num_lateral_classes, device):
    model = MobileNetV1ConvLSTM(
        num_classes=num_classes,
        num_lateral_classes=num_lateral_classes,
        backbone_name=CONFIG["backbone_name"],
        pretrained=CONFIG["pretrained_backbone"],
        projected_channels=CONFIG["feature_channels"],
        hidden_channels=CONFIG["convlstm_hidden_channels"],
        kernel_size=CONFIG["convlstm_kernel_size"],
        dropout=CONFIG["dropout"],
    )
    return model.to(device)


set_seed(CONFIG["seed"])
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
        "found. Attach it to Kaggle or configure additional_dataset_roots."
    )
additional_rows, additional_data = load_additional_rows(
    additional_dataset_dirs, rows, classes
)
if CONFIG["require_july24_data"] and not additional_rows:
    raise RuntimeError("The July-24 dataset supplied no new rows.")
base_frame_count = len(rows)
rows = list(rows) + additional_rows

auto_trim_root = find_auto_trim_root(CONFIG["auto_trim_root"])
if CONFIG["require_auto_trim_data"] and auto_trim_root is None:
    raise FileNotFoundError(
        "Required Todos_cortes_auto_trim/deviation directory was not found."
    )
auto_trim_rows, auto_trim_build = (
    build_auto_trim_rows(
        auto_trim_root,
        rows,
        dataset_dir.name,
        output_dir / "built_trainval_auto_trim" / dataset_dir.name,
        CONFIG["image_size"],
        CONFIG["test_days"],
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
        "The auto-trim source supplied no new non-CAIXA rows after existing "
        "days and clips were excluded."
    )
rows.extend(auto_trim_rows)
classes = class_order(rows)
label_to_index = {label: index for index, label in enumerate(classes)}
lateral_classes = ["left", "right", "none"]
lateral_to_index = {label: index for index, label in enumerate(lateral_classes)}

train_frame_rows, validation_frame_rows, test_frame_rows, clip_counts = split_by_clip(
    rows,
    CONFIG["val_fraction"],
    CONFIG["test_days"],
    CONFIG["seed"],
)
if not train_frame_rows or not validation_frame_rows or not test_frame_rows:
    raise RuntimeError(
        "The combined dataset or one frame split is empty. "
        "Check source_clip dates."
    )

unbalanced_train_sequences, train_sequence_build = build_sequences(
    train_frame_rows,
    CONFIG["sequence_length"],
    CONFIG["sequence_stride"],
    CONFIG["pad_short_clips"],
    CONFIG["sequence_target"],
)
unbalanced_validation_sequences, validation_sequence_build = build_sequences(
    validation_frame_rows,
    CONFIG["sequence_length"],
    CONFIG["sequence_stride"],
    CONFIG["pad_short_clips"],
    CONFIG["sequence_target"],
)
test_sequences, test_sequence_build = build_sequences(
    test_frame_rows,
    CONFIG["sequence_length"],
    CONFIG["sequence_stride"],
    CONFIG["pad_short_clips"],
    CONFIG["sequence_target"],
)
if (
    not unbalanced_train_sequences
    or not unbalanced_validation_sequences
    or not test_sequences
):
    raise RuntimeError(
        "One sequence split is empty. Reduce sequence_length or enable "
        "pad_short_clips."
    )

source_train_sequences, train_balance = balance_background_sequences(
    unbalanced_train_sequences,
    classes,
    CONFIG["balance_background_classes"],
)
validation_sequences, validation_balance = balance_background_sequences(
    unbalanced_validation_sequences,
    classes,
    CONFIG["balance_background_classes"],
)
train_sequences, flip_augmentation = add_left_right_flip_augmentation(
    source_train_sequences,
    classes,
    CONFIG["left_right_flip_augmentation"],
)
saved_dataset = (
    save_dataset_used(
        output_dir,
        {
            "train": train_sequences,
            "validation": validation_sequences,
            "test": test_sequences,
        },
        classes,
    )
    if CONFIG["save_dataset_used"]
    else None
)

print(
    f"{dataset_dir.name}: {len(rows)} frames, {len(classes)} classes | "
    f"train {len(train_sequences)}, val {len(validation_sequences)}, "
    f"test {len(test_sequences)} sequences"
)
print(
    "  excluded compound-CAIXA manifest clips: "
    f"{compound_caixa_exclusion['clips']} clips, "
    f"{compound_caixa_exclusion['source_frames']} source frames"
)
print(
    f"  July-24 data merged: {additional_data['frames']} frames "
    f"from {additional_data['clips']} clips"
)
print(
    f"  auto-trim train/validation build: {auto_trim_build['frames']} "
    f"frames from {auto_trim_build['accepted_clips']} new clips | "
    f"skipped {auto_trim_build['skip_counts']}"
)
for split_name, sequence_build in (
    ("train", train_sequence_build),
    ("validation", validation_sequence_build),
    ("test", test_sequence_build),
):
    print(
        f"  {split_name} sequence build: "
        f"{sequence_build['source_frames']} frames -> "
        f"{sequence_build['sequences']} sequences | "
        f"padded short clips={sequence_build['short_clips_padded']}"
    )
for split_name, balance in (
    ("train", train_balance),
    ("validation", validation_balance),
):
    print(
        f"  {split_name} background balance: "
        f"{balance['background_class']} "
        f"{balance['source_background_sequences']} -> "
        f"{balance['selected_background_sequences']} "
        f"(target {balance['target_background_sequences']}) | "
        f"{balance['source_sequences']} -> "
        f"{balance['effective_sequences']} sequences"
    )
print(
    "  left/right sequence flip augmentation: "
    f"{'applied' if flip_augmentation['applied'] else 'not applicable'} | "
    f"{flip_augmentation['source_train_sequences']} source -> "
    f"{flip_augmentation['effective_train_sequences']} effective sequences"
)
if saved_dataset:
    print(
        f"  saved exact dataset: {saved_dataset['images']} images | "
        f"{saved_dataset['path']}"
    )

# ``datagen.py`` runs this same file with this marker set. Keeping the
# materialization in the trainer's execution path makes the exported sequence
# manifests exactly equivalent to the dataset a normal training run consumes.
if DATA_ONLY:
    print("done | generated exact ConvLSTM dataset; training skipped")
    raise SystemExit(0)

for split_name, split_samples in (
    ("train_before_balance", unbalanced_train_sequences),
    ("train_source", source_train_sequences),
    ("train_effective", train_sequences),
    ("validation_before_balance", unbalanced_validation_sequences),
    ("validation", validation_sequences),
    ("test", test_sequences),
):
    counts = Counter(sample["label"] for sample in split_samples)
    print(
        f"  {split_name}: "
        + ", ".join(
            f"{label}={counts.get(label, 0)}" for label in classes
        )
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
train_loader = make_loader(
    train_sequences,
    label_to_index,
    lateral_to_index,
    transform,
    device,
    True,
    CONFIG["seed"],
)
validation_loader = make_loader(
    validation_sequences,
    label_to_index,
    lateral_to_index,
    transform,
    device,
    False,
    CONFIG["seed"],
)
test_loader = make_loader(
    test_sequences,
    label_to_index,
    lateral_to_index,
    transform,
    device,
    False,
    CONFIG["seed"],
)

model = create_model(len(classes), len(lateral_classes), device)
print(f"  MobileNetV1 backbone selected: {model.backbone_name}")
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"],
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-7
)
scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
best_val_loss = float("inf")
best_state = None
bad_epochs = 0
epochs_completed = 0
history = []

for epoch in range(CONFIG["epochs"]):
    model.train()
    train_loss_sum = train_obstacle_loss_sum = train_lateral_loss_sum = 0.0
    train_correct = train_lateral_correct = 0
    train_examples = 0
    for sequences, labels, lateral_labels in train_loader:
        sequences = sequences.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        lateral_labels = lateral_labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            obstacle_logits, lateral_logits = model(sequences)
            obstacle_loss = F.cross_entropy(obstacle_logits, labels)
            lateral_loss = F.cross_entropy(lateral_logits, lateral_labels)
            loss = obstacle_loss + lateral_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        train_loss_sum += loss.item() * batch_size
        train_obstacle_loss_sum += obstacle_loss.item() * batch_size
        train_lateral_loss_sum += lateral_loss.item() * batch_size
        train_correct += (obstacle_logits.argmax(1) == labels).sum().item()
        train_lateral_correct += (lateral_logits.argmax(1) == lateral_labels).sum().item()
        train_examples += batch_size

    validation = evaluate(model, validation_loader, classes, lateral_classes, device)
    train_loss = train_loss_sum / max(1, train_examples)
    train_accuracy = train_correct / max(1, train_examples)
    scheduler.step(validation["loss"])
    epochs_completed = epoch + 1
    history.append(
        {
            "epoch": epochs_completed,
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_accuracy, 6),
            "train_obstacle_loss": round(train_obstacle_loss_sum / max(1, train_examples), 6),
            "train_lateral_loss": round(train_lateral_loss_sum / max(1, train_examples), 6),
            "train_lateral_accuracy": round(train_lateral_correct / max(1, train_examples), 6),
            "validation_loss": validation["loss"],
            "validation_obstacle": validation["obstacle"],
            "validation_lateral": validation["lateral"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
    )
    print(
        f"epoch {epochs_completed:02d}: train_loss={train_loss:.4f} "
        f"train_acc={train_accuracy:.4f} "
        f"val_loss={validation['loss']:.4f} "
        f"val_obstacle_acc={validation['obstacle']['accuracy']:.4f} "
        f"val_lateral_acc={validation['lateral']['accuracy']:.4f}"
    )

    if validation["loss"] < best_val_loss:
        best_val_loss = validation["loss"]
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
    raise RuntimeError("MobileNetV1 ConvLSTM training produced no checkpoint.")
model.load_state_dict(best_state)
validation = evaluate(model, validation_loader, classes, lateral_classes, device)
test = evaluate(model, test_loader, classes, lateral_classes, device)

checkpoint = {
    "model_state_dict": best_state,
    "architecture": "mobilenetv1_convlstm",
    "backbone_name": model.backbone_name,
    "pretrained_initialization": (
        "imagenet" if CONFIG["pretrained_backbone"] else "random"
    ),
    "classes": classes,
    "lateral_classes": lateral_classes,
    "heads": {"obstacle": classes, "lateral": lateral_classes},
    "image_size": CONFIG["image_size"],
    "sequence_length": CONFIG["sequence_length"],
    "sequence_stride": CONFIG["sequence_stride"],
    "sequence_target": CONFIG["sequence_target"],
    "feature_channels": CONFIG["feature_channels"],
    "convlstm_hidden_channels": CONFIG["convlstm_hidden_channels"],
    "convlstm_kernel_size": CONFIG["convlstm_kernel_size"],
    "normalization": NORMALIZATION,
    "training_augmentation": flip_augmentation,
    "background_balance": {
        "train": train_balance,
        "validation": validation_balance,
    },
    "sequence_build": {
        "train": train_sequence_build,
        "validation": validation_sequence_build,
        "test": test_sequence_build,
    },
    "additional_data": additional_data,
    "auto_trim_build": auto_trim_build,
    "saved_dataset": saved_dataset,
}
torch.save(checkpoint, output_dir / "model.pt")

metrics = {
    "dataset": dataset_dir.name,
    "architecture": "mobilenetv1_convlstm",
    "backbone_name": model.backbone_name,
    "pretrained_initialization": (
        "imagenet" if CONFIG["pretrained_backbone"] else "random"
    ),
    "selection": "clip_split_then_fixed_length_temporal_sequences",
    "excluded_compound_caixa": compound_caixa_exclusion,
    "additional_data": additional_data,
    "auto_trim_build": auto_trim_build,
    "saved_dataset": saved_dataset,
    "training_augmentation": flip_augmentation,
    "background_balance": {
        "train": train_balance,
        "validation": validation_balance,
    },
    "sequence_build": {
        "train": train_sequence_build,
        "validation": validation_sequence_build,
        "test": test_sequence_build,
    },
    "classes": classes,
    "lateral_classes": lateral_classes,
    "heads": {"obstacle": classes, "lateral": lateral_classes},
    "test_days": CONFIG["test_days"],
    "frames": {
        "base": base_frame_count,
        "all": len(rows),
        "july24": len(additional_rows),
        "auto_trim_trainval_pool": len(auto_trim_rows),
        "train": len(train_frame_rows),
        "validation": len(validation_frame_rows),
        "test": len(test_frame_rows),
    },
    "sequences": {
        "train_before_balance": len(unbalanced_train_sequences),
        "train_source": len(source_train_sequences),
        "train": len(train_sequences),
        "validation_before_balance": len(unbalanced_validation_sequences),
        "validation": len(validation_sequences),
        "test": len(test_sequences),
    },
    "clips": clip_counts,
    "epochs_completed": epochs_completed,
    "history": history,
    "validation": validation,
    "test": test,
}
(output_dir / "metrics.json").write_text(
    json.dumps(metrics, indent=2), encoding="utf-8"
)
print(
    f"done | val_obstacle_f1={validation['obstacle']['macro_f1']:.4f} "
    f"val_lateral_f1={validation['lateral']['macro_f1']:.4f} | {output_dir}"
)
