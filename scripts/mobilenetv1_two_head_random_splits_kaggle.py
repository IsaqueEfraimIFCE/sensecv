# -*- coding: utf-8 -*-
"""Kaggle five-run MobileNetV1 comparison on the ConvLSTM datagen export.

It trains five independent seeds for each split policy, using only
``dataset_used/frames.csv`` and ``dataset_used/images`` written by datagen.py:

* ``random_frames``: every frame is independently assigned to train,
  validation, or test. Frames from one clip may be in every split.
* ``random_clips``: clips are randomly assigned as whole units. Every frame
  belonging to a clip stays in that clip's assigned split.

Both policies deliberately replace the fixed day-22/day-24 split. This file
reuses the proven two-head MobileNetV1 training implementation, but replaces
only its dataset loader and split selection. Keep the base trainer beside this
file (or set ``base_trainer`` below) when uploading to Kaggle.
"""

import csv
import json
import re
import runpy
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


CONFIG = {
    "dataset_used_dir": "/kaggle/working/convlstm_mobilenetv1_all_frames/dataset_used",
    "base_trainer": "auto",
    "output_dir": "/kaggle/working/mobilenetv1_two_head_random_splits",
    "seeds": [43, 44, 45, 46, 47],
    "train_fraction": 0.60,
    "validation_fraction": 0.20,
    "image_size": 224,
    "batch_size": 32,
    "epochs": 40,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "patience": 6,
    "num_workers": 2,
    "pretrained": True,
    "hflip_train": True,
    # Leave as None to use every exported frame. Set target_fps=10 for the
    # 10-FPS companion; source_fps records the original-frame assumption.
    "target_fps": None,
    "source_fps": 30.0,
}

OBSTACLE_CLASSES = {"free_path", "obstacle"}
DEVIATION_CLASSES = {"left", "right", "none"}
KAGGLE_BASE = Path(
    "/kaggle/input/datasets/anamariabarrosnonato/scripthelper/"
    "mobilenetv1_two_head_all_frames.py"
)


def base_trainer_path():
    configured = str(CONFIG["base_trainer"])
    if configured.lower() != "auto":
        return Path(configured)
    if KAGGLE_BASE.is_file():
        return KAGGLE_BASE
    if "__file__" in globals():
        return Path(__file__).with_name("mobilenetv1_two_head_all_frames.py")
    return Path.cwd() / "mobilenetv1_two_head_all_frames.py"


def load_all_frames(dataset_used_dir):
    root = Path(dataset_used_dir)
    frames_path = root / "frames.csv"
    if not frames_path.is_file() or not (root / "images").is_dir():
        raise FileNotFoundError(
            f"Expected datagen frames.csv and images/ under {root}"
        )
    rows = []
    with frames_path.open(encoding="utf-8-sig", newline="") as file:
        for frame_index, row in enumerate(csv.DictReader(file)):
            image = row.get("image_filename", "").strip()
            obstacle = row.get("label", "").strip()
            deviation = row.get("lateral_label", "").strip()
            clip = row.get("clip", "").strip()
            path = root / image
            match = re.search(r"__f(\d+)\.(?:jpe?g|png)$", image, re.IGNORECASE)
            if path.is_file() and obstacle and deviation and clip:
                rows.append(
                    {
                        "path": str(path), "clip": clip,
                        "obstacle": obstacle, "deviation": deviation,
                        "frame_index": int(match.group(1)) if match else frame_index,
                        "source_frame_index": int(match.group(1)) if match else None,
                    }
                )
    if not rows:
        raise RuntimeError(f"No usable labelled frames in {frames_path}")
    invalid_obstacle = sorted({r["obstacle"] for r in rows} - OBSTACLE_CLASSES)
    invalid_deviation = sorted({r["deviation"] for r in rows} - DEVIATION_CLASSES)
    if invalid_obstacle or invalid_deviation:
        raise ValueError(
            f"Unexpected labels: obstacle={invalid_obstacle}, deviation={invalid_deviation}"
        )
    return rows


def sample_to_fps(rows, target_fps, source_fps):
    """Keep the first exported frame at each target-FPS time position.

    ``dataset_used`` stores original frame numbers in the copied filename but
    not video timestamps. The caller therefore declares the source FPS; for
    the project all-frame 30-FPS exports, target_fps=10 keeps every third
    chronological source frame.
    """
    if target_fps is None:
        return rows
    target_fps, source_fps = float(target_fps), float(source_fps)
    if not 0 < target_fps <= source_fps:
        raise ValueError("target_fps must be positive and no greater than source_fps.")
    by_clip = defaultdict(list)
    for row in rows:
        if row["source_frame_index"] is None:
            raise ValueError(
                "Cannot infer an original frame number from dataset_used image names."
            )
        by_clip[row["clip"]].append(row)
    interval = source_fps / target_fps
    selected = []
    for clip_rows in by_clip.values():
        clip_rows.sort(key=lambda row: row["source_frame_index"])
        next_frame = float(clip_rows[0]["source_frame_index"])
        for row in clip_rows:
            if row["source_frame_index"] + 1e-9 >= next_frame:
                selected.append(row)
                next_frame += interval
    return selected


def split_sizes(total):
    validation = int(round(total * CONFIG["validation_fraction"]))
    train = int(round(total * CONFIG["train_fraction"]))
    test = total - train - validation
    if min(train, validation, test) < 1:
        raise ValueError("Split fractions leave an empty split.")
    return train, validation, test


def random_frame_split(rows, seed):
    """Randomly distribute individual frames; clip membership is ignored."""
    train_size, validation_size, _ = split_sizes(len(rows))
    order = np.random.default_rng(seed).permutation(len(rows))
    return (
        [rows[index] for index in order[:train_size]],
        [rows[index] for index in order[train_size:train_size + validation_size]],
        [rows[index] for index in order[train_size + validation_size:]],
    )


def random_clip_split(rows, seed):
    """Randomly distribute whole clips; no clip crosses split boundaries."""
    by_clip = defaultdict(list)
    for row in rows:
        by_clip[row["clip"]].append(row)
    clips = sorted(by_clip)
    train_size, validation_size, _ = split_sizes(len(clips))
    order = np.random.default_rng(seed).permutation(len(clips))
    train_clips = set(clips[index] for index in order[:train_size])
    validation_clips = set(
        clips[index] for index in order[train_size:train_size + validation_size]
    )
    test_clips = set(clips[index] for index in order[train_size + validation_size:])
    return (
        [row for row in rows if row["clip"] in train_clips],
        [row for row in rows if row["clip"] in validation_clips],
        [row for row in rows if row["clip"] in test_clips],
    )


def split_summary(splits):
    output = {}
    for name, rows in zip(("train", "validation", "test"), splits):
        output[name] = {
            "frames": len(rows),
            "clips": len({row["clip"] for row in rows}),
            "obstacle": dict(sorted(Counter(row["obstacle"] for row in rows).items())),
            "deviation": dict(sorted(Counter(row["deviation"] for row in rows).items())),
        }
    return output


def require_all_classes(splits, mode, seed):
    for split_name, rows in zip(("train", "validation", "test"), splits):
        missing_obstacle = sorted(OBSTACLE_CLASSES - {r["obstacle"] for r in rows})
        missing_deviation = sorted(DEVIATION_CLASSES - {r["deviation"] for r in rows})
        if missing_obstacle or missing_deviation:
            raise RuntimeError(
                f"{mode} seed {seed} gives {split_name} missing classes: "
                f"obstacle={missing_obstacle}, deviation={missing_deviation}. "
                "Choose another seed or use larger split fractions."
            )


def main():
    if not 0 < CONFIG["train_fraction"] < 1 or not 0 < CONFIG["validation_fraction"] < 1:
        raise ValueError("train_fraction and validation_fraction must be between 0 and 1.")
    if CONFIG["train_fraction"] + CONFIG["validation_fraction"] >= 1:
        raise ValueError("train_fraction + validation_fraction must be less than 1.")

    base_path = base_trainer_path()
    if not base_path.is_file():
        raise FileNotFoundError(
            f"Base trainer not found: {base_path}. Set CONFIG['base_trainer']."
        )
    base = runpy.run_path(str(base_path), run_name="_sensecv_mobilenet_base")
    rows = sample_to_fps(
        load_all_frames(CONFIG["dataset_used_dir"]),
        CONFIG["target_fps"],
        CONFIG["source_fps"],
    )
    output_root = Path(CONFIG["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    transform = base["transforms"].Compose([
        base["transforms"].Resize(
            (CONFIG["image_size"], CONFIG["image_size"]),
            base["transforms"].InterpolationMode.BICUBIC,
        ),
        base["transforms"].ToTensor(),
        base["transforms"].Normalize(base["MEAN"], base["STD"]),
    ])
    device = base["torch"].device(
        "cuda" if base["torch"].cuda.is_available() else "cpu"
    )
    report = {
        "dataset_used_dir": str(Path(CONFIG["dataset_used_dir"]).resolve()),
        "source_frames": len(rows),
        "source_clips": len({row["clip"] for row in rows}),
        "frame_sampling": {
            "target_fps": CONFIG["target_fps"],
            "source_fps_assumption": CONFIG["source_fps"],
        },
        "config": CONFIG,
        "runs": {},
    }
    splitters = {"random_frames": random_frame_split, "random_clips": random_clip_split}
    for mode, splitter in splitters.items():
        for seed in CONFIG["seeds"]:
            splits = splitter(rows, seed)
            require_all_classes(splits, mode, seed)
            run_name = f"{mode}_seed_{seed}"
            run_root = output_root / run_name
            run_root.mkdir(parents=True, exist_ok=True)
            base_config = base["CONFIG"]
            base_config.update({
                "output_dir": str(run_root), "models": ["mobilenetv1"],
                "image_size": CONFIG["image_size"], "batch_size": CONFIG["batch_size"],
                "epochs": CONFIG["epochs"], "learning_rate": CONFIG["learning_rate"],
                "weight_decay": CONFIG["weight_decay"], "patience": CONFIG["patience"],
                "num_workers": CONFIG["num_workers"], "pretrained": CONFIG["pretrained"],
                "hflip_train": CONFIG["hflip_train"], "seed": seed,
            })
            base["seed_everything"](seed)
            print(f"\n=== {run_name} | {split_summary(splits)} ===")
            result = base["train_one"](
                "mobilenetv1", *splits, transform, device, run_root
            )
            details = {"split": split_summary(splits), "result": result}
            (run_root / "split_summary.json").write_text(
                json.dumps(details, indent=2), encoding="utf-8"
            )
            report["runs"][run_name] = details
            (output_root / "summary.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
    print(f"DONE: {len(report['runs'])} MobileNetV1 runs -> {output_root}")


if __name__ == "__main__":
    main()
