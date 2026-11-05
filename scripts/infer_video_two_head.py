"""Run a two-head checkpoint over every frame of a video and dump labelled frames.

    python scripts/infer_video_two_head.py <video.mp4> <out_dir> [checkpoint]

Each decoded frame is written as `<idx>_<obstacle>_<deviation>.jpg` at the
source resolution, and `predictions.csv` records the index, timestamp, both
labels and both softmax confidences.

Preprocessing reproduces the training path exactly: cv2 INTER_AREA down to
224x224 (how the sense datasets were materialized), then ImageNet
normalization. The eval harness' BICUBIC resize is a no-op on an already-224
image, so it is omitted rather than applied twice.
"""
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn

DEFAULT_CK = Path(r"C:\Users\Isaque\Desktop\sense_run4_experiments\checkpoints\dinov2_ultimos1__ultimos1.pt")
OBSTACLE_CLASSES = ["free_path", "obstacle"]
DEVIATION_CLASSES = ["left", "right", "none"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TAMANHO = 224
BATCH = 32


class TwoHead(nn.Module):
    def __init__(self, backbone, dim):
        super().__init__()
        self.backbone = backbone
        self.obstacle_head = nn.Linear(dim, 2)
        self.deviation_head = nn.Linear(dim, 3)

    def forward(self, x):
        f = self.backbone(x)
        return self.obstacle_head(f), self.deviation_head(f)


def load_model(caminho, device):
    ck = torch.load(caminho, map_location="cpu", weights_only=False)
    if ck.get("backbone") == "dinov2_vits14":
        backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    else:
        backbone = timm.create_model(ck["config"]["backbone"], pretrained=False, num_classes=0)
    backbone.eval()
    with torch.inference_mode():
        dim = backbone(torch.zeros(1, 3, 224, 224)).shape[1]
    model = TwoHead(backbone, dim)
    model.load_state_dict(ck["state_dict"])
    return model.eval().to(device)


def to_tensor(frame_bgr):
    pequeno = cv2.resize(frame_bgr, (TAMANHO, TAMANHO), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(pequeno, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1))


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    video = Path(sys.argv[1])
    out = Path(sys.argv[2])
    caminho_ck = Path(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_CK
    if not video.is_file():
        sys.exit(f"missing video: {video}")
    if not caminho_ck.is_file():
        sys.exit(f"missing checkpoint: {caminho_ck}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} | checkpoint={caminho_ck.name}", flush=True)
    model = load_model(caminho_ck, device)

    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    linhas = []
    lote, meta = [], []
    t0 = time.time()

    def flush():
        if not lote:
            return
        x = torch.stack(lote).to(device)
        with torch.inference_mode():
            ol, dl = model(x)
            po = torch.softmax(ol.float(), dim=1).cpu()
            pd = torch.softmax(dl.float(), dim=1).cpu()
        for k, (idx, t, frame) in enumerate(meta):
            io, id_ = int(po[k].argmax()), int(pd[k].argmax())
            obst, desv = OBSTACLE_CLASSES[io], DEVIATION_CLASSES[id_]
            nome = f"{idx:04d}_{obst}_{desv}.jpg"
            if not cv2.imwrite(str(out / nome), frame):
                raise OSError(f"could not write {out / nome}")
            linhas.append({"frame_index": idx, "tempo_s": round(t, 4), "arquivo": nome,
                           "obstacle": obst, "obstacle_conf": round(float(po[k][io]), 4),
                           "deviation": desv, "deviation_conf": round(float(pd[k][id_]), 4),
                           "p_free_path": round(float(po[k][0]), 4),
                           "p_obstacle": round(float(po[k][1]), 4),
                           "p_left": round(float(pd[k][0]), 4),
                           "p_right": round(float(pd[k][1]), 4),
                           "p_none": round(float(pd[k][2]), 4)})
        lote.clear()
        meta.clear()

    idx = 0
    while True:
        pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        ok, frame = cap.read()
        if not ok:
            break
        t = pos if pos > 0 or idx == 0 else idx / fps
        lote.append(to_tensor(frame))
        meta.append((idx, t, frame))
        if len(lote) == BATCH:
            flush()
            print(f"[{time.time()-t0:6.1f}s] {idx + 1} frames", flush=True)
        idx += 1
    flush()
    cap.release()

    with (out / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)

    from collections import Counter
    print(f"\n{len(linhas)} frames in {time.time()-t0:.1f}s -> {out}")
    print("obstacle :", dict(Counter(r["obstacle"] for r in linhas)))
    print("deviation:", dict(Counter(r["deviation"] for r in linhas)))


if __name__ == "__main__":
    main()
