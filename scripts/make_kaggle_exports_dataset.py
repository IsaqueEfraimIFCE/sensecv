# -*- coding: utf-8 -*-
"""Build Kaggle ImageFolder datasets from reviewed SenseCV exports.

This builder is for an interactive ``data/exports`` batch whose export numbers
are not guaranteed to equal the raw clip IDs. It recovers each export's raw
clip by matching its accelerometer samples, validates the export direction
against the collection XLSX manifest, then writes the two established
single-head datasets:

  obstacle_freepath/  -> free_path / obstacle
  deviation_lrn/      -> left / right / none

Every decoded frame is resized to 224x224 by default. Both datasets include
``labels.csv`` and ``dataset_info.json``; the output root also includes
``provenance.csv``. If one raw clip appears more than once, only its first
numbered export is used and later exports are recorded as ignored.

Default invocation builds the 24-07-2026 UFC batch:

    python scripts/make_kaggle_exports_dataset.py --zip
"""

import argparse
import csv
import json
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    "exports": PROJECT_ROOT / "data" / "exports",
    "sources": (
        PROJECT_ROOT
        / "data"
        / "uploaded_datasets"
        / "SenseCV-24-07-2026-UFC"
        / "SenseCV-24-07-2026-UFC"
    ),
    "labels": (
        PROJECT_ROOT
        / "data"
        / "labels"
        / "Coleta - 24 de julho de 2026 - UFC.xlsx"
    ),
    "out": (
        PROJECT_ROOT
        / "data"
        / "derived"
        / "kaggle_24-07-2026_ufc_exports_unique_224"
    ),
    "collection": "SenseCV-24-07-2026-UFC",
    "image_size": 224,
    "jpeg_quality": 95,
}

OBSTACLE_CLASSES = ("free_path", "obstacle")
DEVIATION_CLASSES = ("left", "right", "none")
LABEL_FIELDS = (
    "filename",
    "label",
    "source",
    "event_type",
    "side",
    "source_clip",
    "frame_index",
    "time_s",
    "export_folder",
    "source_id",
    "manifest_obstacle",
)
PROVENANCE_FIELDS = (
    "export_folder",
    "video",
    "source_id",
    "source_clip",
    "event_type",
    "side",
    "obstacle_label",
    "deviation_label",
    "manifest_obstacle_position",
    "manifest_deviation",
    "manifest_obstacle",
    "accelerometer_match_start",
    "frames",
)

SHEET_NS = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
REL_NS = {
    "p": "http://schemas.openxmlformats.org/package/2006/relationships"
}


def normalize_text(value):
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(text.replace("\n", " ").split()).strip().upper()


def column_index(reference):
    letters = re.match(r"[A-Z]+", reference).group()
    result = 0
    for letter in letters:
        result = result * 26 + ord(letter) - 64
    return result - 1


def xlsx_cell_value(cell, shared_strings):
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(
            node.text or "" for node in cell.findall(".//m:t", SHEET_NS)
        )
    value = cell.find("m:v", SHEET_NS)
    if value is None:
        return ""
    raw = value.text or ""
    return shared_strings[int(raw)] if cell_type == "s" else raw


def read_manifest(path):
    """Read the first XLSX table containing an ID header, without Excel."""
    with ZipFile(path) as archive:
        shared_strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared_strings = [
                "".join(node.text or "" for node in item.findall(".//m:t", SHEET_NS))
                for item in root.findall("m:si", SHEET_NS)
            ]

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(
            archive.read("xl/_rels/workbook.xml.rels")
        )
        relationship_map = {
            item.attrib["Id"]: item.attrib["Target"]
            for item in relationships.findall("p:Relationship", REL_NS)
        }

        for sheet in workbook.findall("m:sheets/m:sheet", SHEET_NS):
            relation_id = sheet.attrib[f"{{{SHEET_NS['r']}}}id"]
            target = relationship_map[relation_id]
            target = (
                target.lstrip("/")
                if target.startswith("/")
                else "xl/" + target.lstrip("/")
            )
            xml = ET.fromstring(archive.read(target))
            rows = []
            for row in xml.findall(".//m:sheetData/m:row", SHEET_NS):
                values = {
                    column_index(cell.attrib["r"]): xlsx_cell_value(
                        cell, shared_strings
                    )
                    for cell in row.findall("m:c", SHEET_NS)
                }
                if values:
                    rows.append(values)

            for header_index, row in enumerate(rows):
                headers = {
                    normalize_text(value): column
                    for column, value in row.items()
                }
                if "ID" not in headers:
                    continue
                manifest = {}
                for data_row in rows[header_index + 1 :]:
                    raw_id = str(data_row.get(headers["ID"], "")).strip()
                    if not raw_id:
                        continue
                    try:
                        source_id = f"{int(float(raw_id)):02d}"
                    except ValueError:
                        continue
                    manifest[source_id] = {
                        header: str(data_row.get(column, "")).strip()
                        for header, column in headers.items()
                    }
                if manifest:
                    return manifest
    raise ValueError(f"No manifest table with an ID header found in {path}")


def acceleration_tuple(sample):
    return sample["x"], sample["y"], sample["z"]


def load_source_accelerations(source_root):
    sources = {}
    for folder in sorted(path for path in source_root.iterdir() if path.is_dir()):
        path = folder / "accelerations.json"
        if not path.is_file():
            continue
        samples = json.loads(path.read_text(encoding="utf-8"))["accelerations"]
        starts = defaultdict(list)
        for index, sample in enumerate(samples):
            starts[acceleration_tuple(sample)].append(index)
        sources[folder.name.zfill(2)] = {
            "folder": folder,
            "samples": samples,
            "starts": starts,
        }
    return sources


def recover_source_id(export_folder, sources, probe_count=12):
    path = export_folder / "accelerations.json"
    if not path.is_file():
        raise ValueError(f"{export_folder.name}: accelerations.json not found")
    samples = json.loads(path.read_text(encoding="utf-8"))["accelerations"]
    if not samples:
        raise ValueError(f"{export_folder.name}: empty accelerations.json")
    probes = [acceleration_tuple(sample) for sample in samples[:probe_count]]
    matches = []
    for source_id, source in sources.items():
        source_samples = source["samples"]
        for start in source["starts"].get(probes[0], []):
            if all(
                start + offset < len(source_samples)
                and acceleration_tuple(source_samples[start + offset]) == probe
                for offset, probe in enumerate(probes)
            ):
                matches.append((source_id, start))
    if len(matches) != 1:
        raise ValueError(
            f"{export_folder.name}: expected one accelerometer match, got {matches}"
        )
    return matches[0]


def export_taxonomy(folder_name):
    normalized = normalize_text(folder_name).lower()
    if "desvio_esquerda" in normalized:
        event_type, side = "desvio", "LEFT"
    elif "desvio_direita" in normalized:
        event_type, side = "desvio", "RIGHT"
    elif "parada" in normalized:
        event_type, side = "parada", "NONE"
    elif "sem_obstaculos" in normalized or normalized.endswith("_livre"):
        event_type, side = "livre", "NONE"
    else:
        raise ValueError(f"{folder_name}: unsupported export taxonomy")

    obstacle_position = ""
    match = re.search(r"obstaculo_(direita|esquerda|centro)", normalized)
    if match:
        obstacle_position = match.group(1).upper()

    if event_type == "livre":
        obstacle_label, deviation_label = "free_path", "none"
    elif event_type == "desvio":
        obstacle_label = "obstacle"
        deviation_label = "left" if side == "LEFT" else "right"
    else:
        obstacle_label, deviation_label = "obstacle", "none"
    return {
        "event_type": event_type,
        "side": side,
        "obstacle_position": obstacle_position,
        "obstacle_label": obstacle_label,
        "deviation_label": deviation_label,
    }


def manifest_side(row):
    value = normalize_text(row.get("ORIENTACAO DESVIO", ""))
    if "ESQUERDA" in value:
        return "LEFT"
    if "DIREITA" in value:
        return "RIGHT"
    return "NONE"


def choose_video(folder):
    videos = sorted(folder.glob("*.mp4"))
    if not videos:
        raise ValueError(f"{folder.name}: no MP4 found")
    exact = [video for video in videos if video.stem == folder.name]
    return (exact or videos)[0]


def decode_frames(video_path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        frames.append(
            (
                frame_index,
                frame_index / fps if fps > 0 else 0.0,
                frame,
            )
        )
        frame_index += 1
    capture.release()
    if not frames:
        raise ValueError(f"No decodable frames in {video_path}")
    return frames


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def class_counts(rows, classes):
    counts = Counter(row["label"] for row in rows)
    return {label: counts.get(label, 0) for label in classes}


def build(args):
    exports_root = args.exports.resolve()
    sources_root = args.sources.resolve()
    labels_path = args.labels.resolve()
    output_root = args.out.resolve()
    for path, kind in (
        (exports_root, "exports directory"),
        (sources_root, "sources directory"),
        (labels_path, "labels workbook"),
    ):
        if not path.exists():
            raise SystemExit(f"{kind} not found: {path}")
    if output_root.exists():
        raise SystemExit(f"Output already exists: {output_root}")

    manifest = read_manifest(labels_path)
    sources = load_source_accelerations(sources_root)
    export_folders = sorted(
        path for path in exports_root.iterdir() if path.is_dir()
    )
    if not export_folders:
        raise SystemExit(f"No export folders found under {exports_root}")

    jobs = []
    errors = []
    for folder in export_folders:
        try:
            source_id, match_start = recover_source_id(folder, sources)
            manifest_row = manifest[source_id]
            taxonomy = export_taxonomy(folder.name)
            expected_side = manifest_side(manifest_row)
            expected_position = normalize_text(
                manifest_row.get("POSICAO OBSTACULO", "")
            )
            if taxonomy["side"] != expected_side:
                raise ValueError(
                    f"export side {taxonomy['side']} != manifest {expected_side}"
                )
            if (
                taxonomy["obstacle_position"]
                and expected_position
                and taxonomy["obstacle_position"] != expected_position
            ):
                raise ValueError(
                    "export obstacle position "
                    f"{taxonomy['obstacle_position']} != manifest "
                    f"{expected_position}"
                )
            jobs.append(
                {
                    "folder": folder,
                    "video": choose_video(folder),
                    "source_id": source_id,
                    "match_start": match_start,
                    "manifest": manifest_row,
                    **taxonomy,
                }
            )
        except Exception as error:  # noqa: BLE001
            errors.append(f"{folder.name}: {error}")
    if errors:
        raise SystemExit("Preflight failed:\n  " + "\n  ".join(errors))

    first_job_by_source = {}
    unique_jobs = []
    ignored_duplicate_exports = []
    for job in jobs:
        first_job = first_job_by_source.get(job["source_id"])
        if first_job is not None:
            ignored_duplicate_exports.append(
                {
                    "source_id": job["source_id"],
                    "kept_export": first_job["folder"].name,
                    "ignored_export": job["folder"].name,
                    "reason": "duplicate raw clip; first export wins",
                }
            )
            continue
        first_job_by_source[job["source_id"]] = job
        unique_jobs.append(job)
    jobs = unique_jobs

    dataset_dirs = {
        "obstacle": output_root / "obstacle_freepath",
        "deviation": output_root / "deviation_lrn",
    }
    output_root.mkdir(parents=True)
    for label in OBSTACLE_CLASSES:
        (dataset_dirs["obstacle"] / label).mkdir(parents=True)
    for label in DEVIATION_CLASSES:
        (dataset_dirs["deviation"] / label).mkdir(parents=True)

    rows = {"obstacle": [], "deviation": []}
    provenance = []
    jpeg_options = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
    try:
        for number, job in enumerate(jobs, start=1):
            frames = decode_frames(job["video"])
            source_clip = (
                f"{args.collection}_{args.collection}_{job['source_id']}"
            )
            base = f"exports__{job['folder'].name}"
            manifest_obstacle = job["manifest"].get("OBSTACULO", "")
            for frame_index, time_s, frame in frames:
                if (
                    args.image_size > 0
                    and (
                        frame.shape[0] != args.image_size
                        or frame.shape[1] != args.image_size
                    )
                ):
                    frame = cv2.resize(
                        frame,
                        (args.image_size, args.image_size),
                        interpolation=cv2.INTER_AREA,
                    )
                filename = f"{base}__f{frame_index:05d}.jpg"
                common = {
                    "source": "exports",
                    "event_type": job["event_type"],
                    "side": job["side"],
                    "source_clip": source_clip,
                    "frame_index": frame_index,
                    "time_s": round(time_s, 4),
                    "export_folder": job["folder"].name,
                    "source_id": job["source_id"],
                    "manifest_obstacle": manifest_obstacle,
                }
                for key, label in (
                    ("obstacle", job["obstacle_label"]),
                    ("deviation", job["deviation_label"]),
                ):
                    relative = f"{label}/{filename}"
                    destination = dataset_dirs[key] / relative
                    if not cv2.imwrite(str(destination), frame, jpeg_options):
                        raise OSError(f"Failed to write {destination}")
                    rows[key].append(
                        {"filename": relative, "label": label, **common}
                    )

            provenance.append(
                {
                    "export_folder": job["folder"].name,
                    "video": job["video"].name,
                    "source_id": job["source_id"],
                    "source_clip": source_clip,
                    "event_type": job["event_type"],
                    "side": job["side"],
                    "obstacle_label": job["obstacle_label"],
                    "deviation_label": job["deviation_label"],
                    "manifest_obstacle_position": job["manifest"].get(
                        "POSICAO OBSTACULO", ""
                    ),
                    "manifest_deviation": job["manifest"].get(
                        "ORIENTACAO DESVIO", ""
                    ),
                    "manifest_obstacle": manifest_obstacle,
                    "accelerometer_match_start": job["match_start"],
                    "frames": len(frames),
                }
            )
            print(
                f"[{number:02d}/{len(jobs):02d}] {job['folder'].name} "
                f"-> raw {job['source_id']} | {job['side']} | "
                f"{len(frames)} frames"
            )
    except Exception:
        shutil.rmtree(output_root)
        raise

    write_csv(output_root / "provenance.csv", PROVENANCE_FIELDS, provenance)
    if ignored_duplicate_exports:
        (output_root / "ignored_exports.json").write_text(
            json.dumps(
                ignored_duplicate_exports, indent=2, ensure_ascii=False
            ),
            encoding="utf-8",
        )
    source_ids = sorted({row["source_id"] for row in provenance})
    missing_manifest_ids = sorted(set(manifest) - set(source_ids))
    sources_payload = {
        "exports": str(exports_root),
        "raw_clips": str(sources_root),
        "manifest": str(labels_path),
    }
    summary = {}
    for key, name, classes, task in (
        (
            "obstacle",
            "obstacle_freepath",
            OBSTACLE_CLASSES,
            "obstacle-vs-free-path",
        ),
        (
            "deviation",
            "deviation_lrn",
            DEVIATION_CLASSES,
            "deviation-left-right-none",
        ),
    ):
        write_csv(dataset_dirs[key] / "labels.csv", LABEL_FIELDS, rows[key])
        info = {
            "dataset": name,
            "task": task,
            "classes": list(classes),
            "total_images": len(rows[key]),
            "class_counts": class_counts(rows[key], classes),
            "export_clips": len(provenance),
            "unique_source_clips": len(source_ids),
            "source_ids": source_ids,
            "missing_manifest_source_ids": missing_manifest_ids,
            "ignored_duplicate_exports": ignored_duplicate_exports,
            "frame_selection": (
                f"all decoded export frames, resized to "
                f"{args.image_size}x{args.image_size}"
            ),
            "provenance_method": (
                "exact sequential accelerometer sample match; export taxonomy "
                "validated against XLSX manifest"
            ),
            "sources": sources_payload,
        }
        (dataset_dirs[key] / "dataset_info.json").write_text(
            json.dumps(info, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        summary[name] = {
            "images": len(rows[key]),
            "class_counts": info["class_counts"],
        }

    build_info = {
        "collection": args.collection,
        "image_size": args.image_size,
        "jpeg_quality": args.jpeg_quality,
        "exports": len(provenance),
        "unique_source_clips": len(source_ids),
        "missing_manifest_source_ids": missing_manifest_ids,
        "ignored_duplicate_exports": ignored_duplicate_exports,
        "datasets": summary,
        "sources": sources_payload,
    }
    (output_root / "build_info.json").write_text(
        json.dumps(build_info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    archive_path = None
    if args.zip:
        archive_path = Path(
            shutil.make_archive(
                str(output_root),
                "zip",
                root_dir=output_root.parent,
                base_dir=output_root.name,
            )
        )

    print("\n=== Done ===")
    print(
        f"Exports: {len(provenance)} | unique raw clips: {len(source_ids)} | "
        f"ignored duplicates: {len(ignored_duplicate_exports)} | "
        f"missing raw IDs: {','.join(missing_manifest_ids) or '-'}"
    )
    for name, result in summary.items():
        print(
            f"{name}: {result['images']} images "
            f"{json.dumps(result['class_counts'])}"
        )
    print(f"Output: {output_root}")
    if archive_path:
        print(f"Archive: {archive_path}")
    return output_root, archive_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", type=Path, default=CONFIG["exports"])
    parser.add_argument("--sources", type=Path, default=CONFIG["sources"])
    parser.add_argument("--labels", type=Path, default=CONFIG["labels"])
    parser.add_argument("--out", type=Path, default=CONFIG["out"])
    parser.add_argument("--collection", default=CONFIG["collection"])
    parser.add_argument("--image-size", type=int, default=CONFIG["image_size"])
    parser.add_argument(
        "--jpeg-quality", type=int, default=CONFIG["jpeg_quality"]
    )
    parser.add_argument(
        "--zip", action="store_true", help="Also create <output>.zip"
    )
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
