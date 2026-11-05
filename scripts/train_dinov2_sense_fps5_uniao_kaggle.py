# -*- coding: utf-8 -*-
"""DINOv2 ViT-S/14 two-head training on the 5 FPS uniao configuration.

Same data path as the fps5/uniao MobileNetV4 run: every split of both datasets
subsampled to 5 FPS (source 30), training on distintos+janela, validating on
distintos+janela, so the only change is the backbone and what is unfrozen.

Two runs:
  ultimos2 - only the last 2 transformer blocks + final norm + heads train
  completo - the whole backbone trains

Everything else is run4's configuration: 224x224, Resize(BICUBIC) + ToTensor +
ImageNet Normalize, batch 32, AdamW lr 1e-4 / wd 1e-4, ReduceLROnPlateau(min,
0.5, patience 2, min_lr 1e-7), fp16 AMP, 40 epochs, early stop patience 6, best
state by validation loss, seed 43, deterministic hflip on train only, and
loss = CE(obstacle) + CE(deviation) on every frame (no per-head masking).
"""
import csv
import json
import os
import random
import subprocess
import sys
import tarfile
import time
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

CONFIG = {
    "image_size": 224,          # 16 * 14, valid for ViT-S/14
    "batch_size": 32,
    "epochs": 40,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "patience": 6,
    "seed": 43,
    "num_workers": 2,
    "hflip_train": True,
    "runs": ["ultimos2", "completo"],
    "blocos_destravados": 2,
    "target_fps": 5.0,
    "source_fps": 30.0,
    "input_root": "/kaggle/input",
    "work_root": "/kaggle/working",
    "output_dir": "/kaggle/working/dinov2_fps5_uniao",
}

OBSTACLE_CLASSES = ["free_path", "obstacle"]
DEVIATION_CLASSES = ["left", "right", "none"]
OBS_FROM = {0: "free_path", 1: "obstacle"}
DEV_FROM = {0: "left", 1: "right", 2: "none"}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
SWAP = {"left": "right", "right": "left", "none": "none"}
DISTINTOS = "sense_run4_quadros_distintos_224"
JANELA = "sense_run4_desvio_janela_1_5s_224"


def ensure_compatible_torch():
    if not torch.cuda.is_available():
        return
    try:
        major, minor = torch.cuda.get_device_capability(0)
        nome = torch.cuda.get_device_name(0)
        arquiteturas = torch.cuda.get_arch_list()
    except Exception as erro:  # noqa: BLE001
        print("could not query the GPU:", erro, flush=True)
        return
    alvo = f"sm_{major}{minor}"
    print(f"GPU: {nome} ({alvo}) | torch arch list: {arquiteturas}", flush=True)
    if alvo in arquiteturas:
        return
    if os.environ.get("SENSECV_TORCH_FIXED") == "1":
        raise RuntimeError(f"{nome} ({alvo}) still unsupported after reinstalling torch")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "torch==2.5.1", "torchvision==0.20.1",
                    "--index-url", "https://download.pytorch.org/whl/cu121"], check=True)
    os.environ["SENSECV_TORCH_FIXED"] = "1"
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
            b = int(r["ordem"] * CONFIG["target_fps"] / CONFIG["source_fps"])
            if b not in visto:
                visto.add(b)
                saida.append(r)
    return saida


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


class Frames(Dataset):
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        with Image.open(r["path"]) as im:
            im = im.convert("RGB")
            if r.get("flip"):
                im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            x = self.transform(im)
        return x, OBSTACLE_CLASSES.index(r["obstacle"]), DEVIATION_CLASSES.index(r["deviation"])


class TwoHead(nn.Module):
    def __init__(self, backbone, dim):
        super().__init__()
        self.backbone = backbone
        self.obstacle_head = nn.Linear(dim, 2)
        self.deviation_head = nn.Linear(dim, 3)

    def forward(self, x):
        f = self.backbone(x)
        return self.obstacle_head(f), self.deviation_head(f)


def make_model(estrategia, device):
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone = backbone.to(device)
    backbone.eval()
    with torch.inference_mode():
        dim = backbone(torch.zeros(1, 3, CONFIG["image_size"], CONFIG["image_size"], device=device)).shape[1]
    backbone.train()

    if estrategia == "ultimos2":
        for p in backbone.parameters():
            p.requires_grad = False
        treinaveis = list(backbone.blocks[-CONFIG["blocos_destravados"]:])
        for bloco in treinaveis:
            for p in bloco.parameters():
                p.requires_grad = True
        if hasattr(backbone, "norm"):
            for p in backbone.norm.parameters():
                p.requires_grad = True
    elif estrategia == "completo":
        for p in backbone.parameters():
            p.requires_grad = True
    else:
        raise ValueError(estrategia)

    model = TwoHead(backbone, dim).to(device)
    total = sum(p.numel() for p in model.parameters())
    treinavel = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{estrategia}] trainable {treinavel:,} / {total:,} params "
          f"({100*treinavel/total:.1f}%), feature dim {dim}, blocks {len(backbone.blocks)}", flush=True)
    return model, {"total_params": total, "trainable_params": treinavel,
                   "feature_dim": dim, "blocks": len(backbone.blocks)}


def loader(rows, transform, device, shuffle):
    return DataLoader(Frames(rows, transform), batch_size=CONFIG["batch_size"], shuffle=shuffle,
                      num_workers=CONFIG["num_workers"], pin_memory=device.type == "cuda",
                      generator=torch.Generator().manual_seed(CONFIG["seed"]) if shuffle else None)


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


def evaluate(model, data_loader, device):
    model.eval()
    total, loss_sum, joint = 0, 0.0, 0
    truth, pred = [[], []], [[], []]
    with torch.inference_mode():
        for x, o, d in data_loader:
            x, o, d = x.to(device), o.to(device), d.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                ol, dl = model(x)
                loss = F.cross_entropy(ol, o) + F.cross_entropy(dl, d)
            total += len(x)
            loss_sum += loss.item() * len(x)
            op, dp = ol.argmax(1), dl.argmax(1)
            joint += int(((op == o) & (dp == d)).sum().item())
            truth[0] += o.cpu().tolist(); truth[1] += d.cpu().tolist()
            pred[0] += op.cpu().tolist(); pred[1] += dp.cpu().tolist()
    return {"loss": round(loss_sum / max(1, total), 6),
            "obstacle": metrics(truth[0], pred[0], OBSTACLE_CLASSES),
            "deviation": metrics(truth[1], pred[1], DEVIATION_CLASSES),
            "joint_accuracy": round(joint / max(1, total), 6)}


def count_labels(rows):
    return {"frames": len(rows), "clips": len({r["clip"] for r in rows}),
            "obstacle": dict(Counter(r["obstacle"] for r in rows)),
            "deviation": dict(Counter(r["deviation"] for r in rows)),
            "origem": dict(Counter(r["origem"] for r in rows))}


def train_one(estrategia, treino, validacao, testes, device, saida):
    seed_everything(CONFIG["seed"])
    transform = transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"]), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    efetivo = augment(treino)
    train_loader = loader(efetivo, transform, device, True)
    val_loader = loader(validacao, transform, device, False)
    model, info_modelo = make_model(estrategia, device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5,
                                                           patience=2, min_lr=1e-7)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_loss, best_state, bad, history = float("inf"), None, 0, []
    inicio = time.time()
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        somado = exemplos = 0
        for x, o, d in train_loader:
            x, o, d = x.to(device), o.to(device), d.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                ol, dl = model(x)
                loss = F.cross_entropy(ol, o) + F.cross_entropy(dl, d)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            somado += loss.item() * len(x)
            exemplos += len(x)
        val = evaluate(model, val_loader, device)
        scheduler.step(val["loss"])
        history.append({"epoch": epoch, "train_loss": round(somado / max(1, exemplos), 6),
                        "learning_rate": optimizer.param_groups[0]["lr"], "validation": val})
        print(f"[{estrategia}] epoch {epoch:02d}: train={history[-1]['train_loss']:.4f} "
              f"val={val['loss']:.4f} val_dev_f1={val['deviation']['macro_f1_presentes']:.4f}", flush=True)
        if val["loss"] < best_loss:
            best_loss, best_state, bad = val["loss"], deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= CONFIG["patience"]:
                print(f"[{estrategia}] early stop at epoch {epoch}", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    melhor = min(history, key=lambda e: e["validation"]["loss"]) if history else None
    resultados = {rotulo: evaluate(model, loader(linhas, transform, device, False), device)
                  for rotulo, linhas in testes.items()}
    pasta = Path(saida) / estrategia
    pasta.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": CONFIG, "estrategia": estrategia,
                "backbone": "dinov2_vits14", "modelo": info_modelo,
                "obstacle_classes": OBSTACLE_CLASSES, "deviation_classes": DEVIATION_CLASSES,
                "normalization": {"mean": MEAN, "std": STD}, "image_size": CONFIG["image_size"]},
               pasta / "model.pt")
    final = {"estrategia": estrategia, "backbone": "dinov2_vits14", "modelo": info_modelo,
             "epochs_completed": len(history), "best_epoch": melhor["epoch"] if melhor else None,
             "best_validation_loss": round(best_loss, 6),
             "minutes": round((time.time() - inicio) / 60, 2),
             "splits": {"train_source": count_labels(treino), "train_effective": count_labels(efetivo),
                        "validation": count_labels(validacao),
                        **{f"test_{k}": count_labels(v) for k, v in testes.items()}},
             "test": resultados, "history": history, "config": CONFIG}
    (pasta / "final_statistics.json").write_text(json.dumps(final, indent=2, ensure_ascii=False),
                                                 encoding="utf-8")
    print(f"[{estrategia}] done in {final['minutes']} min", flush=True)
    return final


def main():
    ensure_compatible_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} | torch {torch.__version__}", flush=True)
    distintos = find_dataset(DISTINTOS)
    janela = find_dataset(JANELA)

    bruto = {(o, s): load_split(r, s, o)
             for o, r in (("distintos", distintos), ("janela", janela))
             for s in ("treino", "validacao", "teste")}
    teste_t1 = list(bruto[("distintos", "teste")])
    teste_t4 = teste_t1 + list(bruto[("janela", "teste")])
    dados = {k: sample_to_fps(v) for k, v in bruto.items()}
    for k in bruto:
        print(f"  {k}: {len(bruto[k])} -> {len(dados[k])} after 5 FPS", flush=True)

    testes = {
        "T2_fps5_distintos": dados[("distintos", "teste")],
        "T1_original_distintos": teste_t1,
        "janela_fps5": dados[("janela", "teste")],
        "T4_original_distintos_janela": teste_t4,
    }
    treino = dados[("distintos", "treino")] + dados[("janela", "treino")]
    validacao = dados[("distintos", "validacao")] + dados[("janela", "validacao")]
    print(f"train {len(treino)} | validation {len(validacao)}", flush=True)

    saida = Path(CONFIG["output_dir"])
    saida.mkdir(parents=True, exist_ok=True)
    resumo = {}
    for estrategia in CONFIG["runs"]:
        resumo[estrategia] = train_one(estrategia, treino, validacao, testes, device, saida)

    comparacao = {
        e: {"epochs": r["epochs_completed"], "best_epoch": r["best_epoch"],
            "best_val_loss": r["best_validation_loss"],
            "trainable_params": r["modelo"]["trainable_params"],
            "minutes": r["minutes"],
            **{f"{k}_obstacle_f1": v["obstacle"]["macro_f1_presentes"] for k, v in r["test"].items()},
            **{f"{k}_deviation_f1": v["deviation"]["macro_f1_presentes"] for k, v in r["test"].items()},
            **{f"{k}_joint": v["joint_accuracy"] for k, v in r["test"].items()}}
        for e, r in resumo.items()}
    (saida / "summary.json").write_text(json.dumps(comparacao, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(comparacao, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
