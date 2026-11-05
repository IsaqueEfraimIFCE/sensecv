"""Evaluate every trained checkpoint against the FULL (unsubsampled) test sets.

Both test sets are used untouched, whatever sampling the model was trained with:
  distintos_completo - 3970 frames, every frame of the 118 run4 test clips
  janela_completo    - 2706 frames, every frame of the 60 extended-window clips
"""
import csv
import json
import time
from collections import Counter
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

TMP = Path(r"C:\Users\Isaque\Desktop\sense_run4_experiments")
DESKTOP = Path(r"C:\Users\Isaque\Desktop")
DISTINTOS = DESKTOP / "sense_run4_quadros_distintos_224"
JANELA = DESKTOP / "sense_run4_desvio_janela_1_5s_224"
OBSTACLE_CLASSES = ["free_path", "obstacle"]
DEVIATION_CLASSES = ["left", "right", "none"]
OBSTACLE_FROM_CODE = {0: "free_path", 1: "obstacle"}
DEVIATION_FROM_CODE = {0: "left", 1: "right", 2: "none"}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

CHECKPOINTS = {
    "full_rate/baseline": TMP / "checkpoints/full_rate__baseline.pt",
    "full_rate/uniao": TMP / "checkpoints/full_rate__uniao.pt",
    "fps5/baseline": TMP / "checkpoints/fps5__baseline.pt",
    "fps5/uniao": TMP / "checkpoints/fps5__uniao.pt",
    "dino098/baseline": TMP / "checkpoints/dino098__baseline.pt",
    "dino098/uniao": TMP / "checkpoints/dino098__uniao.pt",
    "dia_validacao/baseline": TMP / "checkpoints/dia_validacao__baseline.pt",
    "dia_validacao/uniao": TMP / "checkpoints/dia_validacao__uniao.pt",
}


def load_split(raiz, split):
    pasta = raiz / split
    linhas = []
    for linha in (pasta / "labels.txt").read_text(encoding="utf-8").splitlines():
        if not linha.strip():
            continue
        nome, obstaculo, desvio = linha.split()
        linhas.append({"path": str(pasta / "images_x_y_224" / nome),
                       "obstacle": OBSTACLE_FROM_CODE[int(obstaculo)],
                       "deviation": DEVIATION_FROM_CODE[int(desvio)]})
    return linhas


class Frames(Dataset):
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        with Image.open(r["path"]) as im:
            x = self.transform(im.convert("RGB"))
        return x, OBSTACLE_CLASSES.index(r["obstacle"]), DEVIATION_CLASSES.index(r["deviation"])


class TwoHead(nn.Module):
    def __init__(self, backbone, feature_dim):
        super().__init__()
        self.backbone = backbone
        self.obstacle_head = nn.Linear(feature_dim, 2)
        self.deviation_head = nn.Linear(feature_dim, 3)

    def forward(self, x):
        f = self.backbone(x)
        return self.obstacle_head(f), self.deviation_head(f)


def metrics(truth, pred, classes):
    n = len(classes)
    m = [[0] * n for _ in range(n)]
    for t, p in zip(truth, pred):
        m[t][p] += 1
    total = sum(map(sum, m))
    f1s, presentes = [], []
    for i in range(n):
        prec = m[i][i] / max(1, sum(m[k][i] for k in range(n)))
        rec = m[i][i] / max(1, sum(m[i]))
        f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        f1s.append(f1)
        if sum(m[i]):
            presentes.append(f1)
    return {"accuracy": round(sum(m[i][i] for i in range(n)) / max(1, total), 6),
            "macro_f1_classes_presentes": round(sum(presentes) / max(1, len(presentes)), 6),
            "per_class_f1": {c: round(f, 6) for c, f in zip(classes, f1s)},
            "support": {c: sum(m[i]) for i, c in enumerate(classes)},
            "confusion_matrix": m}


def evaluate(model, rows, transform, device):
    loader = DataLoader(Frames(rows, transform), batch_size=64, shuffle=False, num_workers=0)
    truth, pred, joint = [[], []], [[], []], 0
    model.eval()
    with torch.inference_mode():
        for x, o, d in loader:
            x = x.to(device)
            ol, dl = model(x)
            op, dp = ol.argmax(1).cpu(), dl.argmax(1).cpu()
            joint += int(((op == o) & (dp == d)).sum())
            truth[0] += o.tolist(); truth[1] += d.tolist()
            pred[0] += op.tolist(); pred[1] += dp.tolist()
    return {"frames": len(rows),
            "obstacle": metrics(truth[0], pred[0], OBSTACLE_CLASSES),
            "deviation": metrics(truth[1], pred[1], DEVIATION_CLASSES),
            "joint_accuracy": round(joint / max(1, len(rows)), 6)}


def main():
    device = torch.device("cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    testes = {"distintos_completo": load_split(DISTINTOS, "teste"),
              "janela_completo": load_split(JANELA, "teste")}
    print({k: len(v) for k, v in testes.items()}, flush=True)

    resultados = {}
    for nome, caminho in CHECKPOINTS.items():
        if not caminho.is_file():
            print(f"skip {nome}: checkpoint not downloaded yet", flush=True)
            continue
        ck = torch.load(caminho, map_location="cpu", weights_only=False)
        backbone = timm.create_model(ck["config"]["backbone"], pretrained=False, num_classes=0)
        backbone.eval()
        with torch.inference_mode():
            dim = backbone(torch.zeros(1, 3, 224, 224)).shape[1]
        model = TwoHead(backbone, dim)
        model.load_state_dict(ck["state_dict"])
        model.to(device)
        resultados[nome] = {}
        for rotulo, linhas in testes.items():
            t0 = time.time()
            resultados[nome][rotulo] = evaluate(model, linhas, transform, device)
            print(f"{nome} on {rotulo}: "
                  f"obs={resultados[nome][rotulo]['obstacle']['macro_f1_classes_presentes']:.4f} "
                  f"dev={resultados[nome][rotulo]['deviation']['macro_f1_classes_presentes']:.4f} "
                  f"joint={resultados[nome][rotulo]['joint_accuracy']:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        (TMP / "resultados" / "eval_full_testsets.json").write_text(
            json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== FULL TEST SETS ===")
    cab = f"{'model':26s} {'distintos obs':>13s} {'distintos dev':>13s} {'dist joint':>10s} {'janela obs':>10s} {'janela dev':>10s} {'jan joint':>9s}"
    print(cab)
    for nome, r in resultados.items():
        d, j = r["distintos_completo"], r["janela_completo"]
        print(f"{nome:26s} {d['obstacle']['macro_f1_classes_presentes']:13.4f} "
              f"{d['deviation']['macro_f1_classes_presentes']:13.4f} {d['joint_accuracy']:10.4f} "
              f"{j['obstacle']['macro_f1_classes_presentes']:10.4f} "
              f"{j['deviation']['macro_f1_classes_presentes']:10.4f} {j['joint_accuracy']:9.4f}")


if __name__ == "__main__":
    main()
