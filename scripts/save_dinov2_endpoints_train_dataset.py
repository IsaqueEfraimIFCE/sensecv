# -*- coding: utf-8 -*-
"""Save the training subset used by train_dinov2_endpoints_kaggle.py.

Paste this file into one Kaggle cell or upload and run it. The script reproduces
the endpoint trainer's current data rules exactly:

* group rows by decoded sequence and keep only its first and last frame;
* keep every source_clip in exactly one split;
* use collection day 22 as the fixed test set;
* treat every undated clear/sem_obstaculos clip as a day-22 clip;
* stratify the remaining clips into train/validation with seed 43.

Only the resulting TRAIN rows are copied. The output is an ImageFolder-style
dataset with class directories, labels.csv, and split_summary.json.
"""

import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


# Keep these split values equal to train_dinov2_endpoints_kaggle.py.
CONFIG = {
    "dataset_dir": (
        "/kaggle/input/datasets/isaqueefraim/dataset8/"
        "kaggle_allframes_datasets_224/obstacle_freepath"
    ),
    "output_dir": "/kaggle/working/dinov2_endpoints_train_dataset",
    "val_fraction": 0.15,
    "test_day": 22,
    "seed": 43,
    # False prevents an accidental replacement. Set True to rebuild this output.
    "overwrite": False,
}


DATE_RE = re.compile(r"(\d{2})-(\d{2})-(\d{4})")
FRAME_RE = re.compile(r"__f(\d+)$", re.IGNORECASE)
CANONICAL_CLASS_ORDERS = (
    ["free_path", "obstacle"],
    ["left", "right", "none"],
)


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
    """Read labels and select the same endpoints as the training script."""
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
                    "filename": filename,
                    "label": label,
                    "clip": clip,
                    "sequence": sequence_name(filename),
                    "frame": frame_number(source_row),
                    "source": source_row.get("source", ""),
                    "event_type": source_row.get("event_type", ""),
                    "side": source_row.get("side", ""),
                    "segment": source_row.get("segment", ""),
                    "time_s": source_row.get("time_s", ""),
                }
                sequences[row["sequence"]].append(row)

    endpoints = []
    for rows in sequences.values():
        rows.sort(key=lambda row: (row["frame"], row["path"]))
        first = dict(rows[0])
        first["endpoint"] = "first"
        endpoints.append(first)
        if rows[-1]["path"] != rows[0]["path"]:
            last = dict(rows[-1])
            last["endpoint"] = "last"
            endpoints.append(last)
        else:
            first["endpoint"] = "first_last"
    return endpoints


def class_order(rows):
    present = {row["label"] for row in rows}
    for order in CANONICAL_CLASS_ORDERS:
        if present == set(order):
            return order
    return sorted(present)


def split_by_clip(rows, val_fraction, test_day, seed):
    """Exact split policy from train_dinov2_endpoints_kaggle.py."""
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


dataset_dir = Path(CONFIG["dataset_dir"])
output_dir = Path(CONFIG["output_dir"])
if dataset_dir.resolve() == output_dir.resolve():
    raise ValueError("output_dir must be different from dataset_dir")
if output_dir.exists():
    if not CONFIG["overwrite"]:
        raise FileExistsError(
            f"Output already exists: {output_dir}. "
            "Choose another output_dir or set overwrite=True."
        )
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True)

rows = read_endpoint_rows(dataset_dir)
classes = class_order(rows)
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

split_rows = {
    "train": train_rows,
    "validation": val_rows,
    "test": test_rows,
}
class_counts = {
    split_name: {
        label: Counter(row["label"] for row in selected).get(label, 0)
        for label in classes
    }
    for split_name, selected in split_rows.items()
}

fields = [
    "filename",
    "label",
    "source_clip",
    "sequence",
    "endpoint",
    "frame_index",
    "time_s",
    "source",
    "event_type",
    "side",
    "segment",
]
saved_rows = []
for row in train_rows:
    relative = Path(row["filename"])
    destination = output_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(row["path"], destination)
    saved_rows.append(
        {
            "filename": relative.as_posix(),
            "label": row["label"],
            "source_clip": row["clip"],
            "sequence": row["sequence"],
            "endpoint": row["endpoint"],
            "frame_index": row["frame"],
            "time_s": row["time_s"],
            "source": row["source"],
            "event_type": row["event_type"],
            "side": row["side"],
            "segment": row["segment"],
        }
    )

with (output_dir / "labels.csv").open(
    "w", encoding="utf-8", newline=""
) as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(saved_rows)

summary = {
    "source_dataset": str(dataset_dir),
    "saved_split": "train",
    "selection": "first_and_last_frame_per_sequence",
    "classes": classes,
    "split_config": {
        "val_fraction": CONFIG["val_fraction"],
        "test_day": CONFIG["test_day"],
        "seed": CONFIG["seed"],
        "undated_clear_day": "22-06-2026",
    },
    "frames": {
        "all_endpoints": len(rows),
        "train": len(train_rows),
        "validation": len(val_rows),
        "test": len(test_rows),
    },
    "clips": clip_counts,
    "class_counts": class_counts,
}
(output_dir / "split_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)

print(f"Saved {len(saved_rows)} training images to {output_dir}")
for split_name in ("train", "validation", "test"):
    counts = class_counts[split_name]
    print(
        f"  {split_name}: "
        + ", ".join(f"{label}={counts[label]}" for label in classes)
    )
