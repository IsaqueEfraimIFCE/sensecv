# -*- coding: utf-8 -*-
"""Materialize the current DINOv2 source pool as train_val and test.

The script is standalone and intentionally does not perform a train/validation
split or training augmentation. Output layout:

    <output_dir>/
      train_val/images/*.jpg
      test/images/*.jpg
      labels.csv
      split_summary.json
      labels.csv
      split_summary.json

Current policy:
* merge the base all-frame dataset, reviewed July-24 package, and new auto-trim
  clips;
* exclude compound-CAIXA clips and collection day 9;
* put explicitly dated days 22 and 24 in test;
* put every other eligible original frame in train_val;
* store each image once with obstacle_label and deviation_label columns;
* do not balance, subsample, flip, or split train from validation.

Paste this whole file into one Kaggle cell, edit ``CONFIG``, and run it.
"""

import csv
import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


CONFIG = {
    "datasets_root": (
        "/kaggle/input/datasets/isaqueefraim/dataset8/"
        "kaggle_allframes_datasets_224"
    ),
    "output_dir": (
        "/kaggle/working/dinov2_allframes_multilabel_trainval_test"
    ),
    "additional_dataset_roots": ["auto"],
    "require_july24_data": True,
    "auto_trim_root": "auto",
    "require_auto_trim_data": True,
    "image_size": 224,
    "excluded_days": [9],
    "test_days": [22, 24],
    # False prevents accidentally replacing a completed export.
    "overwrite": False,
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
    clips = {row["clip"] for row in excluded_rows}
    return kept_rows, {
        "days": sorted(excluded_days),
        "frames": len(excluded_rows),
        "clips": len(clips),
        "source_clips": sorted(clips),
    }


def read_training_rows(dataset_dir):
    labels_path = dataset_dir / "labels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(f"labels.csv not found: {labels_path}")
    rows = []
    excluded_clips = set()
    excluded_frames = 0
    with labels_path.open(encoding="utf-8-sig", newline="") as file:
        for source_row in csv.DictReader(file):
            image_path = dataset_dir / source_row.get("filename", "")
            label = source_row.get("label", "").strip()
            clip = source_row.get("source_clip", "").strip()
            if is_compound_caixa_clip(clip):
                excluded_clips.add(clip)
                excluded_frames += 1
                continue
            if image_path.is_file() and label and clip:
                rows.append(
                    {"path": str(image_path), "label": label, "clip": clip}
                )
    return rows, {
        "clips": len(excluded_clips),
        "source_frames": excluded_frames,
        "source_clips": sorted(excluded_clips),
    }


def class_order(rows):
    present = {row["label"] for row in rows}
    for order in CANONICAL_CLASS_ORDERS:
        if present == set(order):
            return order
    return sorted(present)


def find_additional_dataset_dirs(primary_dataset_dir, configured_roots):
    task_name = primary_dataset_dir.name
    package_roots = []
    explicit_dirs = []
    for value in configured_roots or []:
        if str(value).strip().lower() == "auto":
            package_roots.extend(
                [
                    Path.cwd()
                    / "data"
                    / "derived"
                    / ADDITIONAL_DATASET_FOLDER,
                    primary_dataset_dir.parent / ADDITIONAL_DATASET_FOLDER,
                ]
            )
            kaggle_input = Path("/kaggle/input")
            if kaggle_input.is_dir():
                package_roots.extend(
                    path
                    for path in kaggle_input.rglob(
                        ADDITIONAL_DATASET_FOLDER
                    )
                    if path.is_dir()
                )
                for marker in kaggle_input.rglob("build_info.json"):
                    try:
                        info = json.loads(
                            marker.read_text(encoding="utf-8")
                        )
                    except Exception:
                        continue
                    if (
                        info.get("collection")
                        == "SenseCV-24-07-2026-UFC"
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
    skipped_overlap = set()
    sources = []
    for dataset_dir in dataset_dirs:
        rows, exclusions = read_training_rows(dataset_dir)
        unknown = sorted({row["label"] for row in rows} - set(classes))
        if unknown:
            raise ValueError(
                f"{dataset_dir}: labels not in primary dataset: {unknown}"
            )
        rows_by_clip = defaultdict(list)
        for row in rows:
            rows_by_clip[row["clip"]].append(row)
        source_accepted = 0
        for clip, clip_rows in rows_by_clip.items():
            if clip in base_clips or clip in accepted_clips:
                skipped_overlap.add(clip)
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
        "skipped_overlap_clips": sorted(skipped_overlap),
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
        / AUTO_TRIM_RELATIVE
    ]
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
            "OpenCV is required to build the auto-trim data."
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


def validate_output_path(dataset_dir, output_dir):
    dataset_dir = dataset_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir == dataset_dir or dataset_dir in output_dir.parents:
        raise ValueError("output_dir must be outside dataset_dir")
    if output_dir == Path.cwd().resolve() or len(output_dir.parts) < 3:
        raise ValueError(f"Refusing unsafe output_dir: {output_dir}")


def destination_name(row):
    source_path = Path(row["path"])
    digest = hashlib.sha1(
        (
            f"{row['clip']}|{source_path.name}"
        ).encode("utf-8")
    ).hexdigest()[:12]
    safe_clip = re.sub(r"[^A-Za-z0-9._-]+", "_", row["clip"]).strip("_")
    return f"{safe_clip}__{digest}__{source_path.name}"


def pair_task_rows(obstacle_dir, deviation_dir):
    obstacle_rows, obstacle_exclusion = read_training_rows(obstacle_dir)
    deviation_rows, deviation_exclusion = read_training_rows(deviation_dir)

    def keyed(rows):
        result = {}
        for row in rows:
            key = (row["clip"], Path(row["path"]).name)
            if key in result:
                raise RuntimeError(f"Duplicate frame identity: {key}")
            result[key] = row
        return result

    obstacle_by_key = keyed(obstacle_rows)
    deviation_by_key = keyed(deviation_rows)
    obstacle_keys = set(obstacle_by_key)
    deviation_keys = set(deviation_by_key)
    if obstacle_keys != deviation_keys:
        raise RuntimeError(
            f"Task manifests do not pair one-to-one: "
            f"obstacle-only={len(obstacle_keys - deviation_keys)}, "
            f"deviation-only={len(deviation_keys - obstacle_keys)}"
        )
    rows = []
    for key in sorted(obstacle_keys):
        obstacle = obstacle_by_key[key]
        deviation = deviation_by_key[key]
        rows.append(
            {
                "path": obstacle["path"],
                "clip": obstacle["clip"],
                "obstacle_label": obstacle["label"],
                "deviation_label": deviation["label"],
            }
        )
    return rows, {
        "obstacle_freepath": obstacle_exclusion,
        "deviation_lrn": deviation_exclusion,
    }


def find_additional_multilabel_roots(datasets_root, configured_roots):
    obstacle_dirs = find_additional_dataset_dirs(
        datasets_root / "obstacle_freepath", configured_roots
    )
    deviation_dirs = find_additional_dataset_dirs(
        datasets_root / "deviation_lrn", configured_roots
    )
    deviation_by_root = {path.parent.resolve(): path for path in deviation_dirs}
    roots = []
    for obstacle_dir in obstacle_dirs:
        root = obstacle_dir.parent.resolve()
        if root not in deviation_by_root:
            raise FileNotFoundError(
                f"Missing deviation_lrn beside {obstacle_dir}"
            )
        roots.append(root)
    return roots


def load_additional_multilabel_rows(package_roots, base_rows):
    base_clips = {row["clip"] for row in base_rows}
    accepted_clips = set()
    accepted_rows = []
    skipped_overlap = set()
    sources = []
    for root in package_roots:
        rows, exclusions = pair_task_rows(
            root / "obstacle_freepath", root / "deviation_lrn"
        )
        rows_by_clip = defaultdict(list)
        for row in rows:
            rows_by_clip[row["clip"]].append(row)
        accepted_count = 0
        for clip, clip_rows in rows_by_clip.items():
            if clip in base_clips or clip in accepted_clips:
                skipped_overlap.add(clip)
                continue
            accepted_clips.add(clip)
            accepted_rows.extend(clip_rows)
            accepted_count += len(clip_rows)
        sources.append(
            {
                "dataset_root": str(root),
                "available_frames": len(rows),
                "accepted_frames": accepted_count,
                "excluded_compound_caixa": exclusions,
            }
        )
    return accepted_rows, {
        "dataset_roots": [str(root) for root in package_roots],
        "frames": len(accepted_rows),
        "clips": len(accepted_clips),
        "skipped_overlap_clips": sorted(skipped_overlap),
        "sources": sources,
    }


def add_deviation_labels_to_auto_rows(auto_rows, labels_path):
    with labels_path.open(encoding="utf-8-sig", newline="") as file:
        metadata = list(csv.DictReader(file))
    if len(auto_rows) != len(metadata):
        raise RuntimeError(
            "Auto-trim frame rows and metadata have different lengths."
        )
    combined = []
    for row, meta in zip(auto_rows, metadata):
        deviation_label = auto_trim_label(
            "deviation_lrn",
            meta.get("event_type", ""),
            meta.get("side", ""),
        )
        if deviation_label is None:
            raise RuntimeError(
                f"Unable to derive deviation label for {row['clip']}"
            )
        combined.append(
            {
                "path": row["path"],
                "clip": row["clip"],
                "obstacle_label": row["label"],
                "deviation_label": deviation_label,
            }
        )
    return combined


def write_dataset(output_dir, split_rows):
    saved_rows = []
    seen_destinations = set()
    for split_name in ("train_val", "test"):
        (output_dir / split_name / "images").mkdir(
            parents=True, exist_ok=True
        )
        for row in split_rows[split_name]:
            relative = (
                Path(split_name)
                / "images"
                / destination_name(row)
            )
            if relative in seen_destinations:
                raise RuntimeError(f"Duplicate destination: {relative}")
            seen_destinations.add(relative)
            destination = output_dir / relative
            shutil.copy2(row["path"], destination)
            saved_rows.append(
                {
                    "filename": relative.as_posix(),
                    "split": split_name,
                    "obstacle_label": row["obstacle_label"],
                    "deviation_label": row["deviation_label"],
                    "source_clip": row["clip"],
                    "source_path": row["path"],
                }
            )
    return saved_rows


datasets_root = Path(CONFIG["datasets_root"])
output_dir = Path(CONFIG["output_dir"])
validate_output_path(datasets_root, output_dir)
for task_name in ("obstacle_freepath", "deviation_lrn"):
    if not (datasets_root / task_name / "labels.csv").is_file():
        raise FileNotFoundError(
            f"Task dataset not found: {datasets_root / task_name}"
        )
if output_dir.exists():
    if not CONFIG["overwrite"]:
        raise FileExistsError(
            f"Output already exists: {output_dir}. Set overwrite=True to "
            "replace the previous one."
        )
    shutil.rmtree(output_dir)
output_dir.parent.mkdir(parents=True, exist_ok=True)
auto_trim_root = find_auto_trim_root(CONFIG["auto_trim_root"])
if CONFIG["require_auto_trim_data"] and auto_trim_root is None:
    raise FileNotFoundError(
        "Required Todos_cortes_auto_trim/deviation source not found."
    )

base_rows, caixa_exclusion = pair_task_rows(
    datasets_root / "obstacle_freepath",
    datasets_root / "deviation_lrn",
)
additional_roots = find_additional_multilabel_roots(
    datasets_root, CONFIG["additional_dataset_roots"]
)
if CONFIG["require_july24_data"] and not additional_roots:
    raise FileNotFoundError("Required paired July-24 package not found.")
additional_rows, additional_data = load_additional_multilabel_rows(
    additional_roots, base_rows
)
if CONFIG["require_july24_data"] and not additional_rows:
    raise RuntimeError("Paired July-24 package supplied no new rows.")
base_frames = len(base_rows)
rows = list(base_rows) + additional_rows
rows, excluded_day_data = exclude_collection_days(
    rows, CONFIG["excluded_days"]
)

with tempfile.TemporaryDirectory(
    prefix="dinov2_auto_trim_", dir=output_dir.parent
) as temporary:
    auto_cache = Path(temporary) / "obstacle_freepath"
    obstacle_auto_rows, auto_build = (
        build_auto_trim_rows(
            auto_trim_root,
            rows,
            "obstacle_freepath",
            auto_cache,
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
    if CONFIG["require_auto_trim_data"] and not obstacle_auto_rows:
        raise RuntimeError("Auto-trim supplied no eligible rows.")
    auto_rows = add_deviation_labels_to_auto_rows(
        obstacle_auto_rows, auto_cache / "labels.csv"
    )
    auto_build["temporary_cache_removed_after_export"] = True
    rows.extend(auto_rows)
    test_days = {int(day) for day in CONFIG["test_days"]}
    split_rows = {
        "train_val": [
            row
            for row in rows
            if collection_day(row["clip"]) not in test_days
        ],
        "test": [
            row
            for row in rows
            if collection_day(row["clip"]) in test_days
        ],
    }
    if not split_rows["train_val"] or not split_rows["test"]:
        raise RuntimeError("train_val or test is empty.")
    output_dir.mkdir(parents=True)
    saved_rows = write_dataset(output_dir, split_rows)

label_fields = [
    "filename",
    "split",
    "obstacle_label",
    "deviation_label",
    "source_clip",
    "source_path",
]
with (output_dir / "labels.csv").open(
    "w", encoding="utf-8", newline=""
) as file:
    writer = csv.DictWriter(file, fieldnames=label_fields)
    writer.writeheader()
    writer.writerows(saved_rows)

class_counts = {}
clip_counts = {}
for split_name, selected in split_rows.items():
    class_counts[split_name] = {
        "obstacle_label": dict(
            sorted(Counter(row["obstacle_label"] for row in selected).items())
        ),
        "deviation_label": dict(
            sorted(Counter(row["deviation_label"] for row in selected).items())
        ),
    }
    clip_counts[split_name] = len({row["clip"] for row in selected})
summary = {
    "source_datasets_root": str(datasets_root),
    "selection": "unique_images_with_two_labels_trainval_vs_fixed_test",
    "label_columns": ["obstacle_label", "deviation_label"],
    "split_config": {
        "excluded_days": CONFIG["excluded_days"],
        "test_days": CONFIG["test_days"],
        "train_validation_split_applied": False,
        "balancing_applied": False,
        "augmentation_applied": False,
    },
    "excluded_compound_caixa": caixa_exclusion,
    "excluded_day_data": excluded_day_data,
    "additional_data": additional_data,
    "auto_trim_build": auto_build,
    "frames": {
        "base": base_frames,
        "july24": len(additional_rows),
        "auto_trim": len(auto_rows),
        "all": len(rows),
        "train_val": len(split_rows["train_val"]),
        "test": len(split_rows["test"]),
        "saved_unique_images": len(saved_rows),
    },
    "clips": clip_counts,
    "class_counts": class_counts,
}
(output_dir / "split_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print(f"Saved {len(saved_rows)} unique images with two labels to {output_dir}")
for split_name in ("train_val", "test"):
    counts = class_counts[split_name]
    print(
        f"  {split_name}: {len(split_rows[split_name])} images, "
        f"{clip_counts[split_name]} clips | "
        f"obstacle={counts['obstacle_label']} | "
        f"deviation={counts['deviation_label']}"
    )
