# -*- coding: utf-8 -*-
"""MobileNetV4 two-head training variants on the sense-style SenseCV datasets.

Base configuration is run4's, unchanged: timm
``mobilenetv4_conv_small.e2400_r224_in1k`` pretrained, 224x224, Resize(BICUBIC)
+ ToTensor + ImageNet Normalize, batch 32, AdamW lr 1e-4 / wd 1e-4,
ReduceLROnPlateau(min, 0.5, patience 2, min_lr 1e-7), fp16 AMP, 40 epochs,
early stop patience 6, best state by validation loss, seed 43, deterministic
horizontal-flip augmentation on train only. Every frame carries both labels and
trains both heads: loss = CE(obstacle) + CE(deviation), no per-head masking.

VARIANTE selects one experiment:
  fps5           - subsample every split of both datasets to 5 FPS (source 30)
  dino098        - DINOv2 ViT-S/14 cosine selection at threshold 0.98, per clip,
                   on every split of both datasets
  dia_validacao  - validation is one held-out collection day (21-06-2026)
                   instead of run4's clip-grouped validation

Each variant trains both configurations:
  baseline - quadros distintos only
  uniao    - quadros distintos + janela estendida
Every model is also scored on the untouched full distinct test set
("distintos_completo") so results stay comparable across variants.
"""
import csv
import json
import os
import random
import re
import subprocess
import sys
import tarfile
import time
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

VARIANTE = "fps5"  # PACKAGER REPLACES THIS LINE

CONFIG = {
    "variante": VARIANTE,
    "backbone": "mobilenetv4_conv_small.e2400_r224_in1k",
    "image_size": 224,
    "batch_size": 32,
    "epochs": 40,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "patience": 6,
    "seed": 43,
    "num_workers": 2,
    "pretrained": True,
    "hflip_train": True,
    "runs": ["baseline", "uniao"],
    "target_fps": 5.0,
    "source_fps": 30.0,
    "dino_threshold": 0.98,
    "dino_batch": 64,
    "validation_day": "21-06-2026",
    "input_root": "/kaggle/input",
    "work_root": "/kaggle/working",
    "output_dir": f"/kaggle/working/mobilenetv4_{VARIANTE}",
}

OBSTACLE_CLASSES = ["free_path", "obstacle"]
DEVIATION_CLASSES = ["left", "right", "none"]
OBSTACLE_FROM_CODE = {0: "free_path", 1: "obstacle"}
DEVIATION_FROM_CODE = {0: "left", 1: "right", 2: "none"}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
SWAP = {"left": "right", "right": "left", "none": "none"}
DISTINTOS = "sense_run4_quadros_distintos_224"
JANELA = "sense_run4_desvio_janela_1_5s_224"
DATE_RE = re.compile(r"(\d{2}-\d{2}-\d{4})")


def ensure_compatible_torch():
    """Kaggle may hand out a Tesla P100 (sm_60) while the preinstalled torch is
    built for sm_70+. Install a torch that covers this GPU and re-exec once."""
    if not torch.cuda.is_available():
        return
    try:
        major, minor = torch.cuda.get_device_capability(0)
        nome = torch.cuda.get_device_name(0)
        arquiteturas = torch.cuda.get_arch_list()
    except Exception as erro:  # noqa: BLE001 - diagnostics only
        print("could not query the GPU:", erro, flush=True)
        return
    alvo = f"sm_{major}{minor}"
    print(f"GPU: {nome} ({alvo}) | torch arch list: {arquiteturas}", flush=True)
    if alvo in arquiteturas:
        return
    if os.environ.get("SENSECV_TORCH_FIXED") == "1":
        raise RuntimeError(f"{nome} ({alvo}) still unsupported after reinstalling torch")
    print(f"{alvo} not supported by this torch build; installing a compatible one", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "torch==2.5.1", "torchvision==0.20.1",
                    "--index-url", "https://download.pytorch.org/whl/cu121"], check=True)
    os.environ["SENSECV_TORCH_FIXED"] = "1"
    print("reexecuting with the reinstalled torch", flush=True)
    os.execv(sys.executable, [sys.executable, os.path.abspath(__file__)] + sys.argv[1:])


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def find_dataset(nome):
    raiz = Path(CONFIG["input_root"])
    for caminho in raiz.rglob(nome):
        if caminho.is_dir() and (caminho / "treino" / "labels.txt").is_file():
            return caminho
    for tar in raiz.rglob(f"{nome}.tar"):
        destino = Path(CONFIG["work_root"]) / "datasets"
        destino.mkdir(parents=True, exist_ok=True)
        if not (destino / nome / "treino" / "labels.txt").is_file():
            with tarfile.open(tar) as t:
                t.extractall(destino)
        return destino / nome
    raise FileNotFoundError(f"dataset {nome} not found under {raiz}")


def load_split(raiz, split, origem):
    pasta = raiz / split
    extra = {}
    caminho_prov = pasta / "proveniencia.csv"
    if caminho_prov.is_file():
        with caminho_prov.open(encoding="utf-8", newline="") as f:
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
        nome, obstaculo, desvio = linha.split()
        clip, ordem = extra.get(nome, (f"{origem}:{split}:{nome}", 0))
        linhas.append({
            "path": str(pasta / "images_x_y_224" / nome),
            "obstacle": OBSTACLE_FROM_CODE[int(obstaculo)],
            "deviation": DEVIATION_FROM_CODE[int(desvio)],
            "clip": clip,
            "ordem": ordem,
            "origem": origem,
        })
    if not linhas:
        raise RuntimeError(f"empty split: {pasta}")
    return linhas


def sample_to_fps(rows):
    """Keep the first frame at each target-FPS time position, per clip."""
    alvo, fonte = CONFIG["target_fps"], CONFIG["source_fps"]
    por_clip = defaultdict(list)
    for r in rows:
        por_clip[r["clip"]].append(r)
    saida = []
    for clip, linhas in por_clip.items():
        linhas.sort(key=lambda r: r["ordem"])
        visto = set()
        for r in linhas:
            bucket = int(r["ordem"] * alvo / fonte)
            if bucket not in visto:
                visto.add(bucket)
                saida.append(r)
    return saida


def carregar_dinov2(device):
    try:
        modelo = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        print("DINOv2 loaded from torch.hub", flush=True)
    except Exception as erro:  # noqa: BLE001
        print("torch.hub failed, falling back to transformers:", erro, flush=True)
        from transformers import AutoModel
        interno = AutoModel.from_pretrained("facebook/dinov2-small")

        class Envolve(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                return self.m(pixel_values=x).last_hidden_state[:, 0]

        modelo = Envolve(interno)
    return modelo.to(device).eval()


def select_dinov2(rows, device, modelo, transform):
    """Per clip: keep a frame when cosine similarity to the last kept one
    drops below the threshold. Same rule as the project's Dinov2Selector."""
    limiar = CONFIG["dino_threshold"]
    por_clip = defaultdict(list)
    for r in rows:
        por_clip[r["clip"]].append(r)
    saida = []
    for clip, linhas in por_clip.items():
        linhas.sort(key=lambda r: r["ordem"])
        vetores = []
        for i in range(0, len(linhas), CONFIG["dino_batch"]):
            lote = linhas[i:i + CONFIG["dino_batch"]]
            imagens = []
            for r in lote:
                with Image.open(r["path"]) as im:
                    imagens.append(transform(im.convert("RGB")))
            x = torch.stack(imagens).to(device)
            with torch.inference_mode():
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=device.type == "cuda"):
                    v = modelo(x)
            v = v.float()
            vetores.append(F.normalize(v, dim=1).cpu())
        vetores = torch.cat(vetores) if vetores else torch.empty(0)
        ultimo = None
        for r, v in zip(linhas, vetores):
            if ultimo is None or float(torch.dot(v, ultimo)) < limiar:
                saida.append(r)
                ultimo = v
    return saida


def dia_do_clip(clip):
    m = DATE_RE.search(clip or "")
    return m.group(1) if m else None


class Frames(Dataset):
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        with Image.open(r["path"]) as im:
            im = im.convert("RGB")
            if r.get("flip"):
                im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            x = self.transform(im)
        return x, OBSTACLE_CLASSES.index(r["obstacle"]), DEVIATION_CLASSES.index(r["deviation"])


class TwoHead(nn.Module):
    def __init__(self, backbone, feature_dim):
        super().__init__()
        self.backbone = backbone
        self.obstacle_head = nn.Linear(feature_dim, 2)
        self.deviation_head = nn.Linear(feature_dim, 3)

    def forward(self, x):
        features = self.backbone(x)
        return self.obstacle_head(features), self.deviation_head(features)


def make_model(device):
    backbone = timm.create_model(CONFIG["backbone"], pretrained=CONFIG["pretrained"], num_classes=0)
    backbone = backbone.to(device)
    backbone.eval()
    with torch.inference_mode():
        exemplo = torch.zeros(1, 3, CONFIG["image_size"], CONFIG["image_size"], device=device)
        features = backbone(exemplo)
    backbone.train()
    if features.ndim != 2:
        raise RuntimeError("backbone must return [batch, features]; got " + str(tuple(features.shape)))
    return TwoHead(backbone, features.shape[1]).to(device)


def augment(rows):
    if not CONFIG["hflip_train"]:
        return list(rows)
    out = list(rows)
    for row in rows:
        flip = dict(row)
        flip["flip"] = True
        flip["deviation"] = SWAP[row["deviation"]]
        out.append(flip)
    return out


def loader(rows, transform, device, shuffle):
    return DataLoader(Frames(rows, transform), batch_size=CONFIG["batch_size"], shuffle=shuffle,
                      num_workers=CONFIG["num_workers"], pin_memory=device.type == "cuda",
                      generator=torch.Generator().manual_seed(CONFIG["seed"]) if shuffle else None)


def metrics(truth, predicted, classes):
    n = len(classes)
    matriz = [[0] * n for _ in range(n)]
    for t, p in zip(truth, predicted):
        matriz[t][p] += 1
    total = sum(map(sum, matriz))
    acertos = sum(matriz[i][i] for i in range(n))
    f1s, presentes = [], []
    for i in range(n):
        tp = matriz[i][i]
        suporte = sum(matriz[i])
        prec = tp / max(1, sum(matriz[k][i] for k in range(n)))
        rec = tp / max(1, suporte)
        f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        f1s.append(f1)
        if suporte:
            presentes.append(f1)
    return {"accuracy": round(acertos / max(1, total), 6),
            "macro_f1": round(sum(f1s) / n, 6),
            "macro_f1_classes_presentes": round(sum(presentes) / max(1, len(presentes)), 6),
            "support": {c: sum(matriz[i]) for i, c in enumerate(classes)},
            "per_class_f1": {c: round(f, 6) for c, f in zip(classes, f1s)},
            "confusion_matrix": matriz}


def evaluate(model, data_loader, device):
    model.eval()
    total, loss_sum, joint_ok = 0, 0.0, 0
    truth, predicted = [[], []], [[], []]
    with torch.inference_mode():
        for x, obstacle, deviation in data_loader:
            x, obstacle, deviation = x.to(device), obstacle.to(device), deviation.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                ologits, dlogits = model(x)
                loss = F.cross_entropy(ologits, obstacle) + F.cross_entropy(dlogits, deviation)
            total += len(x)
            loss_sum += loss.item() * len(x)
            op, dp = ologits.argmax(1), dlogits.argmax(1)
            joint_ok += int(((op == obstacle) & (dp == deviation)).sum().item())
            truth[0] += obstacle.cpu().tolist()
            truth[1] += deviation.cpu().tolist()
            predicted[0] += op.cpu().tolist()
            predicted[1] += dp.cpu().tolist()
    return {"loss": round(loss_sum / max(1, total), 6),
            "obstacle": metrics(truth[0], predicted[0], OBSTACLE_CLASSES),
            "deviation": metrics(truth[1], predicted[1], DEVIATION_CLASSES),
            "joint_accuracy": round(joint_ok / max(1, total), 6)}


def count_labels(rows):
    return {"frames": len(rows), "clips": len({r["clip"] for r in rows}),
            "obstacle": dict(Counter(r["obstacle"] for r in rows)),
            "deviation": dict(Counter(r["deviation"] for r in rows)),
            "origem": dict(Counter(r["origem"] for r in rows))}


def train_one(nome, treino, validacao, testes, device, saida):
    seed_everything(CONFIG["seed"])
    transform = transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"]), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)])
    efetivo = augment(treino)
    train_loader = loader(efetivo, transform, device, True)
    val_loader = loader(validacao, transform, device, False)
    model = make_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"],
                                  weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5,
                                                           patience=2, min_lr=1e-7)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_loss, best_state, bad, history = float("inf"), None, 0, []
    inicio = time.time()
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        somado = exemplos = 0
        for x, obstacle, deviation in train_loader:
            x, obstacle, deviation = x.to(device), obstacle.to(device), deviation.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                ologits, dlogits = model(x)
                loss = F.cross_entropy(ologits, obstacle) + F.cross_entropy(dlogits, deviation)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            somado += loss.item() * len(x)
            exemplos += len(x)
        val = evaluate(model, val_loader, device)
        scheduler.step(val["loss"])
        history.append({"epoch": epoch, "train_loss": round(somado / max(1, exemplos), 6),
                        "learning_rate": optimizer.param_groups[0]["lr"], "validation": val})
        print(f"[{CONFIG['variante']}/{nome}] epoch {epoch:02d}: train={history[-1]['train_loss']:.4f} "
              f"val={val['loss']:.4f} val_dev_f1={val['deviation']['macro_f1']:.4f}", flush=True)
        if val["loss"] < best_loss:
            best_loss, best_state, bad = val["loss"], deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= CONFIG["patience"]:
                print(f"[{CONFIG['variante']}/{nome}] early stop at epoch {epoch}", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    melhor = min(history, key=lambda e: e["validation"]["loss"]) if history else None
    resultados = {rotulo: evaluate(model, loader(linhas, transform, device, False), device)
                  for rotulo, linhas in testes.items()}
    pasta = Path(saida) / nome
    pasta.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": CONFIG,
                "obstacle_classes": OBSTACLE_CLASSES, "deviation_classes": DEVIATION_CLASSES,
                "normalization": {"mean": MEAN, "std": STD}, "image_size": CONFIG["image_size"]},
               pasta / "model.pt")
    final = {
        "variante": CONFIG["variante"],
        "run": nome,
        "architecture": CONFIG["backbone"],
        "epochs_completed": len(history),
        "best_epoch": melhor["epoch"] if melhor else None,
        "best_validation_loss": round(best_loss, 6),
        "minutes": round((time.time() - inicio) / 60, 2),
        "loss": "CE(obstacle) + CE(deviation) em todos os quadros; sem mascara por cabeca",
        "splits": {"train_source": count_labels(treino), "train_effective": count_labels(efetivo),
                   "validation": count_labels(validacao),
                   **{f"test_{k}": count_labels(v) for k, v in testes.items()}},
        "test": resultados,
        "history": history,
        "config": CONFIG,
    }
    (pasta / "final_statistics.json").write_text(json.dumps(final, indent=2, ensure_ascii=False),
                                                 encoding="utf-8")
    print(f"[{CONFIG['variante']}/{nome}] done in {final['minutes']} min", flush=True)
    return final


def main():
    ensure_compatible_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"variante={CONFIG['variante']} | device={device} | torch {torch.__version__} | timm {timm.__version__}",
          flush=True)
    distintos = find_dataset(DISTINTOS)
    janela = find_dataset(JANELA)

    dados = {
        ("distintos", "treino"): load_split(distintos, "treino", "distintos"),
        ("distintos", "validacao"): load_split(distintos, "validacao", "distintos"),
        ("distintos", "teste"): load_split(distintos, "teste", "distintos"),
        ("janela", "treino"): load_split(janela, "treino", "janela"),
        ("janela", "validacao"): load_split(janela, "validacao", "janela"),
        ("janela", "teste"): load_split(janela, "teste", "janela"),
    }
    teste_completo = list(dados[("distintos", "teste")])
    antes = {k: len(v) for k, v in dados.items()}

    transform_sel = transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"]), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)])

    if CONFIG["variante"] == "fps5":
        dados = {k: sample_to_fps(v) for k, v in dados.items()}
    elif CONFIG["variante"] == "dino098":
        modelo = carregar_dinov2(device)
        for k in list(dados):
            t0 = time.time()
            dados[k] = select_dinov2(dados[k], device, modelo, transform_sel)
            print(f"  dinov2 {k}: {antes[k]} -> {len(dados[k])} ({time.time()-t0:.0f}s)", flush=True)
        del modelo
        if device.type == "cuda":
            torch.cuda.empty_cache()
    elif CONFIG["variante"] == "dia_validacao":
        dia = CONFIG["validation_day"]
        for origem in ("distintos", "janela"):
            pool = dados[(origem, "treino")] + dados[(origem, "validacao")]
            dados[(origem, "treino")] = [r for r in pool if dia_do_clip(r["clip"]) != dia]
            dados[(origem, "validacao")] = [r for r in pool if dia_do_clip(r["clip"]) == dia]
            if not dados[(origem, "validacao")]:
                raise RuntimeError(f"no {dia} clips in {origem}")
    else:
        raise ValueError("unknown variante: " + CONFIG["variante"])

    print("split sizes after the variant:", flush=True)
    for k in dados:
        print(f"  {k}: {antes[k]} -> {len(dados[k])}", flush=True)

    for a, b in (("treino", "validacao"), ("treino", "teste"), ("validacao", "teste")):
        for x, y in ((("distintos", a), ("janela", b)), (("distintos", b), ("janela", a)),
                     (("distintos", a), ("distintos", b)), (("janela", a), ("janela", b))):
            cruzados = {r["clip"] for r in dados[x]} & {r["clip"] for r in dados[y]}
            if cruzados:
                raise RuntimeError(f"clip leakage {x} vs {y}: {sorted(cruzados)[:5]}")
    print("no clip leakage between any pair of splits", flush=True)

    testes = {"distintos": dados[("distintos", "teste")],
              "janela": dados[("janela", "teste")],
              "distintos_completo": teste_completo}
    saida = Path(CONFIG["output_dir"])
    saida.mkdir(parents=True, exist_ok=True)
    resumo = {}
    if "baseline" in CONFIG["runs"]:
        resumo["baseline"] = train_one("baseline", dados[("distintos", "treino")],
                                       dados[("distintos", "validacao")], testes, device, saida)
    if "uniao" in CONFIG["runs"]:
        resumo["uniao"] = train_one("uniao",
                                    dados[("distintos", "treino")] + dados[("janela", "treino")],
                                    dados[("distintos", "validacao")] + dados[("janela", "validacao")],
                                    testes, device, saida)

    comparacao = {
        f"{CONFIG['variante']}/{nome}": {
            "epochs": r["epochs_completed"],
            "best_epoch": r["best_epoch"],
            "best_val_loss": r["best_validation_loss"],
            "train_frames": r["splits"]["train_source"]["frames"],
            "validation_frames": r["splits"]["validation"]["frames"],
            **{f"{k}_obstacle_macro_f1_presentes": v["obstacle"]["macro_f1_classes_presentes"]
               for k, v in r["test"].items()},
            **{f"{k}_deviation_macro_f1_presentes": v["deviation"]["macro_f1_classes_presentes"]
               for k, v in r["test"].items()},
            **{f"{k}_joint_accuracy": v["joint_accuracy"] for k, v in r["test"].items()},
        } for nome, r in resumo.items()
    }
    (saida / "summary.json").write_text(json.dumps(comparacao, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(comparacao, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
