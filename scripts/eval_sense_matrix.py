"""Score every checkpoint on six test sets.

Distinct-frames only:
  T1 original      - the run4 test set, every frame (3970)
  T2 fps5          - T1 subsampled to 5 FPS, exactly as the fps5 kernel did
  T3 dino098       - T1 after DINOv2 ViT-S/14 cosine selection at 0.98
Distinct + janela:
  T4 original      - T1 + every janela test frame (3970 + 2706)
  T5 fps5          - T4 subsampled to 5 FPS
  T6 dino098       - T4 after DINOv2 selection

Each model predicts once over the 6676 unique frames; every test set is a
subset of those predictions, so the views are mutually consistent.
"""
import csv
import json
import time
from collections import Counter, defaultdict
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
OBS_FROM = {0: "free_path", 1: "obstacle"}
DEV_FROM = {0: "left", 1: "right", 2: "none"}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TARGET_FPS, SOURCE_FPS, DINO_THRESHOLD = 5.0, 30.0, 0.98

CHECKPOINTS = {
    "full_rate/baseline": TMP / "checkpoints/full_rate__baseline.pt",
    "full_rate/uniao": TMP / "checkpoints/full_rate__uniao.pt",
    "fps5/baseline": TMP / "checkpoints/fps5__baseline.pt",
    "fps5/uniao": TMP / "checkpoints/fps5__uniao.pt",
    "dino098/baseline": TMP / "checkpoints/dino098__baseline.pt",
    "dino098/uniao": TMP / "checkpoints/dino098__uniao.pt",
    "dia_validacao/baseline": TMP / "checkpoints/dia_validacao__baseline.pt",
    "dia_validacao/uniao": TMP / "checkpoints/dia_validacao__uniao.pt",
    "dinov2/ultimos1": TMP / "checkpoints/dinov2_ultimos1__ultimos1.pt",
    "dinov2/ultimos2": TMP / "checkpoints/dinov2__ultimos2.pt",
    "dinov2/completo": TMP / "checkpoints/dinov2__completo.pt",
}


def load_split(raiz, split, origem):
    pasta = raiz / split
    extra = {}
    with (pasta / "proveniencia.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            ordem = r.get("frame_index") or r.get("ordem_na_janela") or "0"
            try:
                ordem = int(ordem)
            except ValueError:
                ordem = 0
            extra[r["arquivo"]] = (r["clip"], ordem)
    linhas = []
    for linha in (pasta / "labels.txt").read_text(encoding="utf-8").splitlines():
        if not linha.strip():
            continue
        nome, o, d = linha.split()
        clip, ordem = extra[nome]
        linhas.append({"path": str(pasta / "images_x_y_224" / nome), "clip": clip, "ordem": ordem,
                       "obstacle": OBS_FROM[int(o)], "deviation": DEV_FROM[int(d)], "origem": origem})
    return linhas


def sample_to_fps(rows):
    por_clip = defaultdict(list)
    for r in rows:
        por_clip[r["clip"]].append(r)
    saida = []
    for linhas in por_clip.values():
        linhas.sort(key=lambda r: r["ordem"])
        visto = set()
        for r in linhas:
            b = int(r["ordem"] * TARGET_FPS / SOURCE_FPS)
            if b not in visto:
                visto.add(b)
                saida.append(r)
    return saida


class Frames(Dataset):
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        with Image.open(self.rows[i]["path"]) as im:
            return self.transform(im.convert("RGB")), i


def select_dinov2_cached(rows, transform, device, cache_name):
    """DINOv2 selection is deterministic; cache the chosen paths on disk."""
    cache = TMP / "selecoes" / f"dino_sel_{cache_name}.json"
    if cache.is_file():
        escolhidos = set(json.loads(cache.read_text(encoding="utf-8")))
        saida = [r for r in rows if r["path"] in escolhidos]
        print(f"  dinov2 {cache_name}: {len(saida)} frames (from cache)", flush=True)
        return saida
    saida = select_dinov2(rows, transform, device)
    cache.write_text(json.dumps([r["path"] for r in saida]), encoding="utf-8")
    return saida


def select_dinov2(rows, transform, device):
    modelo = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
    por_clip = defaultdict(list)
    for r in rows:
        por_clip[r["clip"]].append(r)
    saida = []
    for linhas in por_clip.values():
        linhas.sort(key=lambda r: r["ordem"])
        loader = DataLoader(Frames(linhas, transform), batch_size=64, shuffle=False, num_workers=0)
        vetores = []
        with torch.inference_mode():
            for x, _ in loader:
                vetores.append(F.normalize(modelo(x.to(device)).float(), dim=1).cpu())
        vetores = torch.cat(vetores)
        ultimo = None
        for r, v in zip(linhas, vetores):
            if ultimo is None or float(torch.dot(v, ultimo)) < DINO_THRESHOLD:
                saida.append(r)
                ultimo = v
    del modelo
    return saida


class TwoHead(nn.Module):
    def __init__(self, backbone, dim):
        super().__init__()
        self.backbone = backbone
        self.obstacle_head = nn.Linear(dim, 2)
        self.deviation_head = nn.Linear(dim, 3)

    def forward(self, x):
        f = self.backbone(x)
        return self.obstacle_head(f), self.deviation_head(f)


def metrics(truth, pred, classes):
    n = len(classes)
    m = [[0] * n for _ in range(n)]
    for t, p in zip(truth, pred):
        m[t][p] += 1
    f1s, presentes = [], []
    for i in range(n):
        prec = m[i][i] / max(1, sum(m[k][i] for k in range(n)))
        rec = m[i][i] / max(1, sum(m[i]))
        f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        f1s.append(f1)
        if sum(m[i]):
            presentes.append(f1)
    return {"accuracy": round(sum(m[i][i] for i in range(n)) / max(1, sum(map(sum, m))), 6),
            "macro_f1_presentes": round(sum(presentes) / max(1, len(presentes)), 6),
            "per_class_f1": {c: round(f, 6) for c, f in zip(classes, f1s)},
            "support": {c: sum(m[i]) for i, c in enumerate(classes)},
            "confusion_matrix": m}


def main():
    device = torch.device("cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    dist = load_split(DISTINTOS, "teste", "distintos")
    jan = load_split(JANELA, "teste", "janela")
    todos = dist + jan
    indice = {r["path"]: i for i, r in enumerate(todos)}
    print(f"unique frames: distintos={len(dist)} janela={len(jan)} total={len(todos)}", flush=True)

    print("running DINOv2 selection (downloads ViT-S/14 on first use)...", flush=True)
    t0 = time.time()
    dino_dist = select_dinov2_cached(dist, transform, device, 'distintos')
    dino_todos = select_dinov2_cached(todos, transform, device, 'todos')
    print(f"  dinov2 done in {time.time()-t0:.0f}s", flush=True)

    conjuntos = {
        "T1 original (distintos)": dist,
        "T2 fps5 (distintos)": sample_to_fps(dist),
        "T3 dino098 (distintos)": dino_dist,
        "T4 original (distintos+janela)": todos,
        "T5 fps5 (distintos+janela)": sample_to_fps(todos),
        "T6 dino098 (distintos+janela)": dino_todos,
    }
    for nome, linhas in conjuntos.items():
        print(f"  {nome}: {len(linhas)} frames | obs={Counter(r['obstacle'] for r in linhas)} "
              f"dev={Counter(r['deviation'] for r in linhas)}", flush=True)

    saida_json = TMP / "resultados" / "eval_matrix.json"
    resultados = {}
    if saida_json.is_file():
        anterior = json.loads(saida_json.read_text(encoding="utf-8"))
        if anterior.get("conjuntos") == {k: len(v) for k, v in conjuntos.items()}:
            resultados = anterior.get("resultados", {})
            print(f"resuming: {len(resultados)} models already scored", flush=True)
        else:
            print("test sets changed; recomputing every model", flush=True)

    for nome, caminho in CHECKPOINTS.items():
        if nome in resultados:
            continue
        if not caminho.is_file():
            print(f"skip {nome} (checkpoint missing)", flush=True)
            continue
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
        model.eval().to(device)

        t0 = time.time()
        pred_o = [0] * len(todos)
        pred_d = [0] * len(todos)
        loader = DataLoader(Frames(todos, transform), batch_size=64, shuffle=False, num_workers=0)
        with torch.inference_mode():
            for x, idx in loader:
                ol, dl = model(x.to(device))
                for k, i in enumerate(idx.tolist()):
                    pred_o[i] = int(ol[k].argmax())
                    pred_d[i] = int(dl[k].argmax())
        print(f"{nome}: predicted {len(todos)} frames in {time.time()-t0:.0f}s", flush=True)

        resultados[nome] = {}
        for rotulo, linhas in conjuntos.items():
            ids = [indice[r["path"]] for r in linhas]
            to = [OBSTACLE_CLASSES.index(r["obstacle"]) for r in linhas]
            td = [DEVIATION_CLASSES.index(r["deviation"]) for r in linhas]
            po = [pred_o[i] for i in ids]
            pd = [pred_d[i] for i in ids]
            joint = sum(1 for a, b, c, e in zip(to, td, po, pd) if a == c and b == e)
            resultados[nome][rotulo] = {
                "frames": len(linhas),
                "obstacle": metrics(to, po, OBSTACLE_CLASSES),
                "deviation": metrics(td, pd, DEVIATION_CLASSES),
                "joint_accuracy": round(joint / max(1, len(linhas)), 6)}
        (TMP / "resultados" / "eval_matrix.json").write_text(
            json.dumps({"conjuntos": {k: len(v) for k, v in conjuntos.items()}, "resultados": resultados},
                       indent=2, ensure_ascii=False), encoding="utf-8")

    for rotulo in conjuntos:
        print(f"\n=== {rotulo} ({len(conjuntos[rotulo])} frames) ===")
        print(f"{'model':26s} {'obs F1':>8s} {'dev F1':>8s} {'joint':>8s}")
        for nome, r in sorted(resultados.items(), key=lambda kv: -kv[1][rotulo]["joint_accuracy"]):
            v = r[rotulo]
            print(f"{nome:26s} {v['obstacle']['macro_f1_presentes']:8.4f} "
                  f"{v['deviation']['macro_f1_presentes']:8.4f} {v['joint_accuracy']:8.4f}")


if __name__ == "__main__":
    main()
