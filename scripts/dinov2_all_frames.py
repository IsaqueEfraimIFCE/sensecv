# -*- coding: utf-8 -*-
"""Standalone Kaggle DINOv2 all-frame trainer with obstacle and lateral heads.

The trainer consumes the fixed ``dataset_used`` export produced by the ConvLSTM
run. It reads the image files and the pre-written train/validation/test manifests
directly; it never rebuilds, augments the source data, or recomputes a split.

Every model writes, immediately after it finishes:
  <output>/<model>/model.pt
  <output>/<model>/training_history.json
  <output>/<model>/final_statistics.json
The test split in the manifest is never used for checkpoint selection.
"""
import csv
import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    import timm
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "timm"])
    import timm

# Edit only this block in Kaggle. dataset_used_dir is the fixed ConvLSTM export.
CONFIG = {
    # Dataset8's parent directory containing obstacle_freepath/ and
    # deviation_lrn/.  A task directory itself is also accepted.
    "dataset_used_dir": "/kaggle/working/convlstm_mobilenetv1_all_frames/dataset_used",
    "output_dir": "/kaggle/working/dinov2_all_frames",
    "models": ["dinov2_vits14"],
    "image_size": 224, "batch_size": 32, "epochs": 40, "learning_rate": 1e-4,
    "weight_decay": 1e-4, "val_fraction": 0.20, "patience": 6, "seed": 43,
    "num_workers": 2, "sequence_length": 8, "sequence_stride": 4,
    "hflip_train": True, "pretrained": True, "test_days": [22, 24],
    "require_july24_data": True, "require_auto_trim_data": True,
}
OBSTACLE_CLASSES = ["free_path", "obstacle"]
DEVIATION_CLASSES = ["left", "right", "none"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TIMM_MODELS = {
    "mobilenetv1": "mobilenetv1_100.ra4_e3600_r224_in1k",
    "mobilenetv4": "mobilenetv4_conv_small.e2400_r224_in1k",
    "resnet50": "resnet50.a1_in1k",
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
    "mobilenetv1_convlstm": "mobilenetv1_100.ra4_e3600_r224_in1k",
}
DATE_RE = re.compile(r"(\d{2})-(\d{2})-(\d{4})")
COMPOUND_CAIXA_CLIP_IDS = {
    "02-06-2026": set(range(9, 29)),
    "06-06-2026": {10, 11, 12, 30, 31, 32, 50, 51, 52, 70, 71, 72},
    "10-06-2026": {5, 6, 7, 20, 21, 22, 35, 36, 37, 50, 51, 52},
}

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True

def find_dataset(value):
    if value != "auto":
        root = Path(value).resolve()
        if root.name in {"obstacle_freepath", "deviation_lrn"}: root = root.parent
        return root
    paired = []
    for path in Path("/kaggle/input").rglob("labels.csv"):
        try:
            with path.open(encoding="utf-8-sig", newline="") as f:
                fields = set((csv.DictReader(f).fieldnames or []))
            if path.parent.name == "obstacle_freepath" and (path.parent.parent / "deviation_lrn" / "labels.csv").is_file():
                paired.append(path.parent.parent)
        except OSError: pass
    if not paired: raise RuntimeError("dataset_dir='auto' found no Dataset8 paired obstacle_freepath/deviation_lrn roots.")
    # Dataset8 is the largest paired attachment; July-24 is merged separately.
    def paired_size(root):
        total = 0
        for name in ("obstacle_freepath", "deviation_lrn"):
            with (root / name / "labels.csv").open(encoding="utf-8-sig") as f:
                total += max(0, sum(1 for _ in f) - 1)
        return total
    return max(paired, key=paired_size)

def collection_day(clip):
    match = DATE_RE.search(clip or "")
    return int(match.group(1)) if match else None

def compound_caixa(clip):
    date = DATE_RE.search(clip or ""); last = re.search(r"_(\d+)$", clip or "")
    return bool(date and last and int(last.group(1)) in COMPOUND_CAIXA_CLIP_IDS.get(date.group(0), set()))

def _single_labels(task_dir):
    """Read the exact all-frame labels contract used by resnet50_all_frames."""
    output = []
    with (task_dir / "labels.csv").open(encoding="utf-8-sig", newline="") as f:
        for source in csv.DictReader(f):
            filename, label, clip = source.get("filename", ""), source.get("label", "").strip(), source.get("source_clip", "").strip()
            path = task_dir / filename
            clear = (filename + " " + clip).lower()
            if DATE_RE.search(clip) is None and ("sem_obstaculos" in clear or source.get("source", "").lower().strip() == "clear"):
                clip = "SenseCV-22-06-2026-clear_" + (clip or path.stem)
            if path.is_file() and label and clip and not compound_caixa(clip):
                try: index = int(source.get("frame_index", 0))
                except ValueError: index = 0
                output.append({"path": str(path), "label": label, "clip": clip, "frame_index": index, "key": (clip, path.name)})
    return output

def _pair_task_dirs(root):
    obstacle, deviation = root / "obstacle_freepath", root / "deviation_lrn"
    if not (obstacle / "labels.csv").is_file() or not (deviation / "labels.csv").is_file():
        raise FileNotFoundError("Expected obstacle_freepath/labels.csv and deviation_lrn/labels.csv under " + str(root))
    obstacles = _single_labels(obstacle)
    deviations = {r["key"]: r for r in _single_labels(deviation)}
    rows = []
    for obstacle_row in obstacles:
        deviation_row = deviations.get(obstacle_row["key"])
        if deviation_row is None: continue
        rows.append({"path": obstacle_row["path"], "clip": obstacle_row["clip"], "obstacle": obstacle_row["label"], "deviation": deviation_row["label"], "frame_index": obstacle_row["frame_index"]})
    if not rows: raise RuntimeError("The two task manifests have no matching (source_clip, filename) frames.")
    return rows

def _find_july24_root():
    roots = [p for p in Path("/kaggle/input").rglob("kaggle_24-07-2026_ufc_exports_unique_224") if p.is_dir()]
    return roots[0] if roots else None

def _auto_trim_rows(output, existing):
    """Decode the same eligible auto-trim MP4s once, assigning both labels."""
    roots = [p for p in Path("/kaggle/input").rglob("deviation") if p.is_dir() and p.parent.name == "Todos_cortes_auto_trim"]
    if not roots: return []
    try: import cv2
    except ImportError as e: raise RuntimeError("OpenCV is required for Todos_cortes_auto_trim.") from e
    root, existing_clips = roots[0], {r["clip"] for r in existing}
    existing_dates = {m.group(0) for c in existing_clips if (m := DATE_RE.search(c))}
    ledger = {}
    if (root / "sources.csv").is_file():
        with (root / "sources.csv").open(encoding="utf-8-sig", newline="") as f: ledger = {r.get("output_folder", ""): r for r in csv.DictReader(f)}
    cache = output / "built_trainval_auto_trim"; cache.mkdir(parents=True, exist_ok=True); rows = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        clip = folder.name; date = DATE_RE.search(clip)
        if clip in existing_clips or compound_caixa(clip) or collection_day(clip) in CONFIG["test_days"] or (date and date.group(0) in existing_dates): continue
        info = None
        try: info = json.loads((folder / "cut_info.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError): info = ledger.get(clip)
        event, side = (info or {}).get("event_type", "").lower(), (info or {}).get("side", "").upper()
        obstacle = "obstacle" if event in {"desvio", "parada"} else "free_path" if event == "livre" else None
        deviation = {"LEFT": "left", "RIGHT": "right"}.get(side, "none") if event == "desvio" else "none" if event in {"parada", "livre"} else None
        video = next(iter(sorted(folder.glob("*.mp4"))), None)
        if not obstacle or not deviation or video is None: continue
        capture = cv2.VideoCapture(str(video)); index = 0
        while True:
            ok, frame = capture.read()
            if not ok: break
            frame = cv2.resize(frame, (CONFIG["image_size"], CONFIG["image_size"]), interpolation=cv2.INTER_AREA)
            target = cache / f"{clip}__f{index:05d}.jpg"
            if not cv2.imwrite(str(target), frame): raise OSError("Could not write " + str(target))
            rows.append({"path": str(target), "clip": clip, "obstacle": obstacle, "deviation": deviation, "frame_index": index}); index += 1
        capture.release()
    return rows

def load_allframe_rows(root, output):
    """Reproduce the EfficientNet-B0 all-frame dataset construction."""
    rows = _pair_task_dirs(root)
    july = _find_july24_root()
    if july:
        july_rows = _pair_task_dirs(july); known = {r["clip"] for r in rows}; rows.extend(r for r in july_rows if r["clip"] not in known)
    elif CONFIG["require_july24_data"]: raise FileNotFoundError("Required kaggle_24-07-2026_ufc_exports_unique_224 attachment not found.")
    auto = _auto_trim_rows(output, rows)
    if CONFIG["require_auto_trim_data"] and not auto: raise RuntimeError("No eligible auto-trim rows were created; attach Todos_cortes_auto_trim or set require_auto_trim_data=False.")
    return rows + auto

def grouped_split(rows, fraction, seed):
    """Match EfficientNet-B0: fixed-day test, obstacle-stratified clip split."""
    rng = np.random.default_rng(seed); by_clip = defaultdict(list)
    for row in rows: by_clip[row["clip"]].append(row)
    test_clips = {clip for clip in by_clip if collection_day(clip) in set(CONFIG["test_days"])}
    strata = defaultdict(list)
    for clip, items in by_clip.items():
        if clip not in test_clips: strata[Counter(row["obstacle"] for row in items).most_common(1)[0][0]].append(clip)
    validation_clips = set()
    for clips in strata.values():
        clips = list(clips); rng.shuffle(clips); count = len(clips)
        take = max(1, int(round(count * fraction))) if count > 2 else (1 if count > 1 else 0)
        validation_clips.update(clips[:min(take, max(0, count - 1))])
    train = [r for r in rows if r["clip"] not in test_clips and r["clip"] not in validation_clips]
    validation = [r for r in rows if r["clip"] in validation_clips]
    test = [r for r in rows if r["clip"] in test_clips]
    if not train or not validation or not test: raise RuntimeError("A train/validation/test partition is empty.")
    return train, validation, test

def load_saved_splits(dataset_used_dir):
    """Load the fixed ConvLSTM dataset export; never rebuild or resplit it."""
    root = Path(dataset_used_dir)
    frames_path = root / "frames.csv"
    sequences_dir = root / "sequences"
    if not frames_path.is_file() or not sequences_dir.is_dir():
        raise FileNotFoundError(f"Expected frames.csv and sequences/ under {root}")
    with frames_path.open(encoding="utf-8-sig", newline="") as file:
        frame_rows = {row["image_filename"]: row for row in csv.DictReader(file)}
    result = {}
    for split in ("train", "validation", "test"):
        manifest = sequences_dir / f"{split}.csv"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing fixed split manifest: {manifest}")
        seen, rows = set(), []
        with manifest.open(encoding="utf-8-sig", newline="") as file:
            for sample in csv.DictReader(file):
                image_name = sample["image_filename"]
                if image_name in seen:
                    continue
                seen.add(image_name)
                frame = frame_rows.get(image_name)
                if frame is None:
                    raise ValueError(f"{manifest}: image absent from frames.csv: {image_name}")
                obstacle = frame.get("label", "").strip()
                deviation = frame.get("lateral_label", "").strip()
                if not obstacle or not deviation:
                    raise ValueError("frames.csv must include label and lateral_label; regenerate dataset_used with the two-head ConvLSTM trainer.")
                rows.append({"path": str(root / image_name), "clip": frame["clip"], "obstacle": obstacle, "deviation": deviation, "frame_index": len(rows)})
        if not rows:
            raise RuntimeError(f"Fixed {split} split is empty: {manifest}")
        result[split] = rows
    return result["train"], result["validation"], result["test"]


def augment(rows):
    if not CONFIG["hflip_train"]: return list(rows)
    out = list(rows)
    for row in rows:
        flip = dict(row); flip["flip"] = True
        flip["deviation"] = {"left": "right", "right": "left", "none": "none"}[row["deviation"]]
        out.append(flip)
    return out

def balance_background(rows, label_name, background):
    """The same largest-remainder, evenly-spaced reduction as ResNet-50."""
    groups = defaultdict(list); foreground = []
    for row in rows:
        (groups[row["clip"]] if row[label_name] == background else foreground).append(row)
    available, target = sum(map(len, groups.values())), min(sum(map(len, groups.values())), len(foreground))
    summary = {"head": label_name, "background": background, "source_frames": len(rows), "source_background_frames": available, "target_background_frames": target, "selected_background_frames": target, "clips": {}}
    if available <= target: return list(rows), summary
    quotas, fractions, allocated = {}, [], 0
    for clip, values in sorted(groups.items()):
        exact = len(values) * target / available; quotas[clip] = int(np.floor(exact)); allocated += quotas[clip]; fractions.append((exact - quotas[clip], clip))
    for _, clip in sorted(fractions, key=lambda item: (-item[0], item[1]))[:target - allocated]: quotas[clip] += 1
    selected = set()
    for clip, values in groups.items():
        keep = quotas[clip]
        if keep:
            indices = np.floor((np.arange(keep) + .5) * len(values) / keep).astype(int)
            selected.update(id(values[i]) for i in indices)
        summary["clips"][clip] = {"source_frames": len(values), "selected_frames": keep}
    return [r for r in rows if r[label_name] != background or id(r) in selected], summary

def make_head_balanced_rows(rows):
    """Independently reproduce ResNet's reduction for obstacle and deviation.

    A single shared reduction cannot balance both targets (free_path is always
    none), so samples are tagged with the head whose loss they train.  This
    retains one two-output model while making each head see exactly its own
    ResNet-style reduced data.
    """
    obstacle_rows, obstacle_summary = balance_background(rows, "obstacle", "free_path")
    deviation_rows, deviation_summary = balance_background(rows, "deviation", "none")
    output = []
    for row in obstacle_rows:
        item = dict(row); item["obstacle_active"], item["deviation_active"] = True, False; output.append(item)
    for row in deviation_rows:
        item = dict(row); item["obstacle_active"], item["deviation_active"] = False, True; output.append(item)
    return output, {"obstacle": obstacle_summary, "deviation": deviation_summary, "combined_loss_samples": len(output)}

class Frames(Dataset):
    def __init__(self, rows, transform): self.rows, self.transform = rows, transform
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        r = self.rows[index]
        with Image.open(r["path"]) as im:
            im = im.convert("RGB")
            if r.get("flip"): im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            x = self.transform(im)
        return x, OBSTACLE_CLASSES.index(r["obstacle"]), DEVIATION_CLASSES.index(r["deviation"]), r.get("obstacle_active", True), r.get("deviation_active", True)

def make_sequences(rows):
    by_clip = defaultdict(list)
    for r in rows: by_clip[r["clip"]].append(r)
    samples = []
    length, stride = CONFIG["sequence_length"], CONFIG["sequence_stride"]
    for clip_rows in by_clip.values():
        clip_rows.sort(key=lambda r: r["frame_index"])
        if len(clip_rows) < length: clip_rows += [clip_rows[-1]] * (length - len(clip_rows))
        starts = list(range(0, len(clip_rows) - length + 1, stride))
        if starts[-1] != len(clip_rows) - length: starts.append(len(clip_rows) - length)
        for start in starts: samples.append(clip_rows[start:start + length])
    return samples

class Sequences(Dataset):
    def __init__(self, samples, transform): self.samples, self.transform = samples, transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, index):
        sample = self.samples[index]; tensors = []
        for r in sample:
            with Image.open(r["path"]) as im:
                im = im.convert("RGB")
                if r.get("flip"): im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                tensors.append(self.transform(im))
        last = sample[-1]
        return torch.stack(tensors), OBSTACLE_CLASSES.index(last["obstacle"]), DEVIATION_CLASSES.index(last["deviation"]), last.get("obstacle_active", True), last.get("deviation_active", True)

class TwoHead(nn.Module):
    def __init__(self, backbone, feature_dim):
        super().__init__(); self.backbone = backbone
        self.obstacle_head, self.deviation_head = nn.Linear(feature_dim, 2), nn.Linear(feature_dim, 3)
    def forward(self, x):
        features = self.backbone(x)
        return self.obstacle_head(features), self.deviation_head(features)

class ConvLSTMCell(nn.Module):
    def __init__(self, channels):
        super().__init__(); self.gates = nn.Conv2d(channels * 2, channels * 4, 3, padding=1)
    def forward(self, x, state):
        h, c = state; i, f, o, g = self.gates(torch.cat([x, h], 1)).chunk(4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(c), c

class MobileNetConvLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(TIMM_MODELS["mobilenetv1_convlstm"], pretrained=CONFIG["pretrained"], num_classes=0, global_pool="")
        channels = self.backbone.num_features; self.cell = ConvLSTMCell(channels)
        self.obstacle_head, self.deviation_head = nn.Linear(channels, 2), nn.Linear(channels, 3)
    def forward(self, sequence):
        batch, steps, channels, height, width = sequence.shape
        maps = self.backbone(sequence.reshape(batch * steps, channels, height, width))
        maps = maps.reshape(batch, steps, *maps.shape[1:]); h = c = None
        for step in range(steps):
            x = maps[:, step]
            if h is None: h, c = torch.zeros_like(x), torch.zeros_like(x)
            h, c = self.cell(x, (h, c))
        features = h.mean(dim=(2, 3))
        return self.obstacle_head(features), self.deviation_head(features)

def make_two_head(backbone, device):
    """Build heads from the backbone's actual pooled output width.

    Some timm models (notably MobileNetV4 in current Kaggle images) report a
    ``num_features`` width that differs from the tensor returned by
    ``forward()`` when created with ``num_classes=0``.  Probe the same forward
    path used during training so the heads cannot have a stale width.
    """
    backbone = backbone.to(device); was_training = backbone.training; backbone.eval()
    with torch.inference_mode():
        example = torch.zeros(1, 3, CONFIG["image_size"], CONFIG["image_size"], device=device)
        features = backbone(example)
    backbone.train(was_training)
    if features.ndim != 2:
        raise RuntimeError("Backbone must return pooled [batch, features] tensors; got " + str(tuple(features.shape)))
    return TwoHead(backbone, features.shape[1]).to(device)

def make_model(name, device):
    if name == "mobilenetv1_convlstm": return MobileNetConvLSTM().to(device)
    if name == "dinov2_vits14":
        try: backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        except Exception as error: raise RuntimeError("DINOv2 download failed; enable Kaggle Internet.") from error
        return make_two_head(backbone, device)
    if name not in TIMM_MODELS: raise ValueError("Unknown model: " + name)
    return make_two_head(timm.create_model(TIMM_MODELS[name], pretrained=CONFIG["pretrained"], num_classes=0), device)

def evaluate(model, loader, device):
    model.eval(); total, loss_sum, joint_total, joint_correct = 0, 0.0, 0, 0; truth = [[], []]; predicted = [[], []]
    with torch.inference_mode():
        for x, obstacle, deviation, obstacle_active, deviation_active in loader:
            x, obstacle, deviation = x.to(device), obstacle.to(device), deviation.to(device)
            obstacle_active, deviation_active = obstacle_active.to(device).bool(), deviation_active.to(device).bool()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                ologits, dlogits = model(x); loss = ologits.sum() * 0.0
                if obstacle_active.any(): loss = loss + F.cross_entropy(ologits[obstacle_active], obstacle[obstacle_active])
                if deviation_active.any(): loss = loss + F.cross_entropy(dlogits[deviation_active], deviation[deviation_active])
            total += len(x); loss_sum += loss.item() * len(x)
            truth[0] += obstacle[obstacle_active].cpu().tolist(); truth[1] += deviation[deviation_active].cpu().tolist()
            predicted[0] += ologits.argmax(1)[obstacle_active].cpu().tolist(); predicted[1] += dlogits.argmax(1)[deviation_active].cpu().tolist()
            both = obstacle_active & deviation_active
            joint_total += both.sum().item()
            joint_correct += ((ologits.argmax(1) == obstacle) & (dlogits.argmax(1) == deviation) & both).sum().item()
    def scores(y, p, count):
        matrix = [[sum(a == i and b == j for a, b in zip(y, p)) for j in range(count)] for i in range(count)]
        f1 = []
        for i in range(count):
            tp = matrix[i][i]; precision = tp / max(1, sum(matrix[r][i] for r in range(count))); recall = tp / max(1, sum(matrix[i]))
            f1.append(2 * precision * recall / max(1e-12, precision + recall))
        return {"accuracy": round(sum(a == b for a, b in zip(y, p)) / max(1, len(y)), 6), "macro_f1": round(float(np.mean(f1)), 6), "confusion_matrix": matrix}
    o, d = scores(truth[0], predicted[0], 2), scores(truth[1], predicted[1], 3)
    joint = round(joint_correct / joint_total, 6) if joint_total else None
    return {"loss": round(loss_sum / max(1, total), 6), "obstacle": o, "deviation": d, "joint_accuracy": joint}

def loader(rows, transform, device, sequence, shuffle, seed):
    data = Sequences(make_sequences(rows), transform) if sequence else Frames(rows, transform)
    return DataLoader(data, batch_size=CONFIG["batch_size"], shuffle=shuffle, num_workers=CONFIG["num_workers"], pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(seed) if shuffle else None)

def count_labels(rows):
    return {"frames": len(rows), "clips": len({r["clip"] for r in rows}), "obstacle": dict(Counter(r["obstacle"] for r in rows)), "deviation": dict(Counter(r["deviation"] for r in rows))}

def train_one(name, train, validation, test, transform, device, output_root):
    sequence = name == "mobilenetv1_convlstm"; out = output_root / name; out.mkdir(parents=True, exist_ok=True)
    reduced_train, balance = make_head_balanced_rows(train)
    reduced_validation, validation_balance = make_head_balanced_rows(validation)
    effective_train = augment(reduced_train)
    train_loader = loader(effective_train, transform, device, sequence, True, CONFIG["seed"])
    val_loader = loader(reduced_validation, transform, device, sequence, False, CONFIG["seed"])
    test_loader = loader(test, transform, device, sequence, False, CONFIG["seed"])
    model = make_model(name, device); optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=2, min_lr=1e-7)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda"); best_loss, best_state, bad, history = float("inf"), None, 0, []
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train(); summed = examples = 0
        for x, obstacle, deviation, obstacle_active, deviation_active in train_loader:
            x, obstacle, deviation = x.to(device), obstacle.to(device), deviation.to(device); optimizer.zero_grad(set_to_none=True)
            obstacle_active, deviation_active = obstacle_active.to(device).bool(), deviation_active.to(device).bool()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                ologits, dlogits = model(x)
                loss = ologits.sum() * 0.0
                if obstacle_active.any(): loss = loss + F.cross_entropy(ologits[obstacle_active], obstacle[obstacle_active])
                if deviation_active.any(): loss = loss + F.cross_entropy(dlogits[deviation_active], deviation[deviation_active])
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); summed += loss.item() * len(x); examples += len(x)
        val = evaluate(model, val_loader, device); scheduler.step(val["loss"])
        record = {"epoch": epoch, "train_loss": round(summed / max(1, examples), 6), "learning_rate": optimizer.param_groups[0]["lr"], "validation": val}; history.append(record)
        print(f"{name} epoch {epoch:02d}: train_loss={record['train_loss']:.4f} val_loss={val['loss']:.4f}")
        if val["loss"] < best_loss:
            best_loss, best_state, bad = val["loss"], deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= CONFIG["patience"]: break
    model.load_state_dict(best_state); validation_metrics, test_metrics = evaluate(model, val_loader, device), evaluate(model, test_loader, device)
    checkpoint = {"state_dict": model.state_dict(), "architecture": name, "obstacle_classes": OBSTACLE_CLASSES, "deviation_classes": DEVIATION_CLASSES, "image_size": CONFIG["image_size"], "normalization": {"mean": MEAN, "std": STD}, "sequence_length": CONFIG["sequence_length"] if sequence else None}
    torch.save(checkpoint, out / "model.pt")
    (out / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    final = {"architecture": name, "epochs_completed": len(history), "best_validation_loss": best_loss, "background_reduction": {"train": balance, "validation": validation_balance}, "splits": {"train_source": count_labels(train), "train_reduced_per_head": count_labels(reduced_train), "train_effective": count_labels(effective_train), "validation_source": count_labels(validation), "validation_reduced_per_head": count_labels(reduced_validation), "test": count_labels(test)}, "validation": validation_metrics, "test": test_metrics}
    (out / "final_statistics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(f"DONE {name}: test joint={test_metrics['joint_accuracy']:.4f}; wrote {out}")
    return final

def main():
    seed_everything(CONFIG["seed"])
    dataset_used = Path(CONFIG["dataset_used_dir"])
    output = Path(CONFIG["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    train, validation, test = load_saved_splits(dataset_used)
    transform = transforms.Compose([transforms.Resize((CONFIG["image_size"], CONFIG["image_size"]), transforms.InterpolationMode.BICUBIC), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"dataset_used={dataset_used} device={device} | train={len(train)}, val={len(validation)}, test={len(test)}")
    summary = {name: train_one(name, train, validation, test, transform, device, output) for name in CONFIG["models"]}
    (output / "summary.json").write_text(json.dumps({"config": CONFIG, "dataset_used_dir": str(dataset_used), "models": summary}, indent=2), encoding="utf-8")

if __name__ == "__main__": main()

