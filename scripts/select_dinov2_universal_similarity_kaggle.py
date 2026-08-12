# -*- coding: utf-8 -*-
"""Select diverse frames using universal DINOv2 cosine similarity.

Paste this whole file into one Kaggle cell, edit CONFIG if needed, enable
Internet for torch.hub, and run it.

The first ``max_input_frames`` valid TRAIN_VAL rows are processed in manifest
order. A candidate is selected only when its cosine similarity is strictly
below ``cosine_threshold`` for EVERY previously selected train_val image.
Every TEST row bypasses DINOv2 and is copied unchanged.
"""

import csv
import json
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


CONFIG = {
    # "auto" finds a labels.csv with obstacle_label and deviation_label below
    # /kaggle/input. An explicit dataset root is also accepted.
    "dataset_dir": "auto",
    "output_dir": (
        "/kaggle/working/dinov2_similarity_selected_test_preserved"
    ),
    "max_input_frames": 1000,
    "cosine_threshold": 0.90,
    "dino_model": "dinov2_vits14",
    "image_size": 224,
    "batch_size": 64,
    "num_workers": 2,
    # False protects a completed selection from accidental replacement.
    "overwrite": False,
}


REQUIRED_COLUMNS = {
    "filename",
    "split",
    "obstacle_label",
    "deviation_label",
    "source_clip",
}


def is_multilabel_dataset(root):
    labels_path = root / "labels.csv"
    if not labels_path.is_file():
        return False
    with labels_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return REQUIRED_COLUMNS <= set(reader.fieldnames or [])


def find_dataset_dir(configured):
    if str(configured).strip().lower() != "auto":
        candidate = Path(configured)
        if is_multilabel_dataset(candidate):
            return candidate.resolve()
        raise FileNotFoundError(
            f"Multi-label dataset not found at {candidate}"
        )

    candidates = [
        Path.cwd() / "dinov2_allframes_multilabel_trainval_test"
    ]
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        candidates.extend(
            path.parent for path in kaggle_input.rglob("labels.csv")
        )
    found = []
    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen or not is_multilabel_dataset(resolved):
            continue
        seen.add(resolved)
        found.append(resolved)
    if not found:
        raise FileNotFoundError(
            "No multi-label dataset was found below /kaggle/input. "
            "Set CONFIG['dataset_dir'] explicitly."
        )
    if len(found) > 1:
        raise RuntimeError(
            "Multiple multi-label datasets were found; set dataset_dir "
            "explicitly: " + ", ".join(str(path) for path in found)
        )
    return found[0]


def read_source_rows(dataset_dir, trainval_limit):
    labels_path = dataset_dir / "labels.csv"
    trainval_rows = []
    test_rows = []
    missing_images = 0
    with labels_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                "labels.csv is missing: "
                + ", ".join(sorted(missing_columns))
            )
        for manifest_index, source_row in enumerate(reader):
            image_path = dataset_dir / source_row.get("filename", "")
            if not image_path.is_file():
                missing_images += 1
                continue
            split = source_row.get("split", "").strip()
            if split not in {"train_val", "test"}:
                continue
            row = dict(source_row)
            row["path"] = str(image_path)
            row["manifest_index"] = manifest_index
            if split == "test":
                test_rows.append(row)
            elif len(trainval_rows) < trainval_limit:
                trainval_rows.append(row)
    if not trainval_rows:
        raise RuntimeError("No valid train_val images were found.")
    if not test_rows:
        raise RuntimeError("No valid test images were found.")
    return trainval_rows, test_rows, missing_images


class FrameImages(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(self.rows[index]["path"]) as image:
            return self.transform(image.convert("RGB"))


def extract_embeddings(backbone, rows, transform, device):
    loader = DataLoader(
        FrameImages(rows, transform),
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
    return F.normalize(torch.cat(embeddings), p=2, dim=1)


def universal_cosine_selection(embeddings, threshold):
    """Greedily enforce similarity < threshold against every selected row."""
    selected_indices = []
    audit = []
    for index in range(len(embeddings)):
        if not selected_indices:
            selected_indices.append(index)
            audit.append(
                {
                    "selected": True,
                    "max_similarity": None,
                    "nearest_selected_index": None,
                }
            )
            continue

        similarities = embeddings[selected_indices] @ embeddings[index]
        max_similarity, nearest_position = similarities.max(dim=0)
        nearest_index = selected_indices[int(nearest_position.item())]
        accepted = bool(torch.all(similarities < threshold).item())
        if accepted:
            selected_indices.append(index)
        audit.append(
            {
                "selected": accepted,
                "max_similarity": float(max_similarity.item()),
                "nearest_selected_index": nearest_index,
            }
        )
    return selected_indices, audit


def validate_output_path(dataset_dir, output_dir):
    dataset_dir = dataset_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir == dataset_dir or dataset_dir in output_dir.parents:
        raise ValueError("output_dir must be outside dataset_dir")
    if output_dir == Path.cwd().resolve() or len(output_dir.parts) < 3:
        raise ValueError(f"Refusing unsafe output_dir: {output_dir}")


dataset_dir = find_dataset_dir(CONFIG["dataset_dir"])
output_dir = Path(CONFIG["output_dir"])
validate_output_path(dataset_dir, output_dir)
if output_dir.exists():
    if not CONFIG["overwrite"]:
        raise FileExistsError(
            f"Output already exists: {output_dir}. Set overwrite=True to "
            "replace it."
        )
    shutil.rmtree(output_dir)
output_dir.parent.mkdir(parents=True, exist_ok=True)

trainval_rows, test_rows, missing_images = read_source_rows(
    dataset_dir, int(CONFIG["max_input_frames"])
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
embeddings = extract_embeddings(backbone, trainval_rows, transform, device)
del backbone
if device.type == "cuda":
    torch.cuda.empty_cache()

threshold = float(CONFIG["cosine_threshold"])
if not -1.0 <= threshold <= 1.0:
    raise ValueError("cosine_threshold must be between -1 and 1.")
selected_indices, audit = universal_cosine_selection(embeddings, threshold)

output_dir.mkdir(parents=True)
manifest_fields = [
    "filename",
    "split",
    "obstacle_label",
    "deviation_label",
    "source_clip",
    "source_path",
    "original_manifest_index",
    "selection_rank",
    "max_similarity_when_selected",
]
saved_rows = []
rank_by_index = {
    source_index: rank
    for rank, source_index in enumerate(selected_indices)
}
for index in selected_indices:
    row = trainval_rows[index]
    source_relative = Path(row["filename"])
    destination_relative = (
        Path(row["split"]) / "images" / source_relative.name
    )
    destination = output_dir / destination_relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise RuntimeError(f"Duplicate destination: {destination_relative}")
    shutil.copy2(row["path"], destination)
    saved_rows.append(
        {
            "filename": destination_relative.as_posix(),
            "split": row["split"],
            "obstacle_label": row["obstacle_label"],
            "deviation_label": row["deviation_label"],
            "source_clip": row["source_clip"],
            "source_path": row.get("source_path", row["path"]),
            "original_manifest_index": row["manifest_index"],
            "selection_rank": rank_by_index[index],
            "max_similarity_when_selected": (
                ""
                if audit[index]["max_similarity"] is None
                else round(audit[index]["max_similarity"], 8)
            ),
        }
    )
for row in test_rows:
    source_relative = Path(row["filename"])
    destination_relative = Path("test") / "images" / source_relative.name
    destination = output_dir / destination_relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise RuntimeError(f"Duplicate destination: {destination_relative}")
    shutil.copy2(row["path"], destination)
    saved_rows.append(
        {
            "filename": destination_relative.as_posix(),
            "split": "test",
            "obstacle_label": row["obstacle_label"],
            "deviation_label": row["deviation_label"],
            "source_clip": row["source_clip"],
            "source_path": row.get("source_path", row["path"]),
            "original_manifest_index": row["manifest_index"],
            "selection_rank": "",
            "max_similarity_when_selected": "",
        }
    )

with (output_dir / "labels.csv").open(
    "w", encoding="utf-8", newline=""
) as file:
    writer = csv.DictWriter(file, fieldnames=manifest_fields)
    writer.writeheader()
    writer.writerows(saved_rows)

audit_fields = [
    "input_index",
    "original_manifest_index",
    "filename",
    "selected",
    "max_similarity",
    "nearest_selected_input_index",
    "nearest_selected_filename",
]
audit_rows = []
for index, (row, decision) in enumerate(zip(trainval_rows, audit)):
    nearest_index = decision["nearest_selected_index"]
    audit_rows.append(
        {
            "input_index": index,
            "original_manifest_index": row["manifest_index"],
            "filename": row["filename"],
            "selected": decision["selected"],
            "max_similarity": (
                ""
                if decision["max_similarity"] is None
                else round(decision["max_similarity"], 8)
            ),
            "nearest_selected_input_index": (
                "" if nearest_index is None else nearest_index
            ),
            "nearest_selected_filename": (
                ""
                if nearest_index is None
                else trainval_rows[nearest_index]["filename"]
            ),
        }
    )
with (output_dir / "selection_audit.csv").open(
    "w", encoding="utf-8", newline=""
) as file:
    writer = csv.DictWriter(file, fieldnames=audit_fields)
    writer.writeheader()
    writer.writerows(audit_rows)

selected_embeddings = embeddings[selected_indices]
if len(selected_indices) > 1:
    similarity_matrix = selected_embeddings @ selected_embeddings.T
    similarity_matrix.fill_diagonal_(-1.0)
    final_max_similarity = float(similarity_matrix.max().item())
else:
    final_max_similarity = None
if (
    final_max_similarity is not None
    and not final_max_similarity < threshold
):
    raise AssertionError(
        "Final selected set violates the universal cosine threshold."
    )

summary = {
    "source_dataset": str(dataset_dir),
    "selection": "greedy_universal_dinov2_cosine_diversity",
    "dino_model": CONFIG["dino_model"],
    "max_input_frames": CONFIG["max_input_frames"],
    "processed_trainval_frames": len(trainval_rows),
    "test_frames_bypassing_dinov2": len(test_rows),
    "missing_images_skipped_before_limit": missing_images,
    "cosine_rule": (
        "trainval candidate similarity to every selected trainval image "
        "< threshold"
    ),
    "cosine_threshold": threshold,
    "selected_trainval_frames": len(selected_indices),
    "rejected_trainval_frames": (
        len(trainval_rows) - len(selected_indices)
    ),
    "preserved_test_frames": len(test_rows),
    "total_output_frames": len(saved_rows),
    "dinov2_applied_to_test": False,
    "final_max_pairwise_similarity": final_max_similarity,
    "pairwise_constraint_verified": True,
    "device": str(device),
}
(output_dir / "selection_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)

print(
    f"Processed {len(trainval_rows)} train_val frames with DINOv2; "
    f"selected {len(selected_indices)}, rejected "
    f"{len(trainval_rows) - len(selected_indices)}"
)
print(f"Preserved all {len(test_rows)} test frames without DINOv2")
print(
    f"Universal cosine rule verified: every selected pair < {threshold:.4f}"
)
print(f"Saved selected dataset to {output_dir}")
