"""Score every checkpoint on the 10-frame janela variant, against the 0.5 s one.

Test sets (all built from the *teste* split only):
  D     - distinct frames, the run4 test set (3970)          [= T1 of eval_sense_matrix]
  J15   - janela 0.5 s tail, every frame
  J10   - janela 10-frame tail, every frame
  U15   - D + J15                                            [= T4 of eval_sense_matrix]
  U10   - D + J10
Plus the pre/post-cut halves of each janela set, which is where the two tails
actually differ: the cut portion is the same frames, re-decoded.

Each model predicts once over the union of all unique frames; every test set is
a subset of those predictions, so the views are mutually consistent.

Metric definitions and the model wrapper are imported from eval_sense_matrix so
the numbers stay comparable to the existing eval_matrix.json.
"""
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import timm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_sense_matrix as base

TMP = base.TMP
DESKTOP = base.DESKTOP
DISTINTOS = base.DISTINTOS
JANELA_15 = DESKTOP / "sense_run4_desvio_janela_1_5s_224"
JANELA_10 = DESKTOP / "sense_run4_desvio_janela_10q_224"
SAIDA = TMP / "resultados" / "eval_janela10.json"


def pos_corte_map(raiz, split="teste"):
    """arquivo -> pos_corte flag, so a janela set can be split pre/post cut."""
    caminho = raiz / split / "proveniencia.csv"
    with caminho.open(encoding="utf-8", newline="") as f:
        return {r["arquivo"]: int(r["pos_corte"]) for r in csv.DictReader(f)}


def marcar_pos_corte(linhas, raiz):
    flags = pos_corte_map(raiz)
    for r in linhas:
        r["pos_corte"] = flags[Path(r["path"]).name]
    return linhas


def main():
    if not JANELA_10.is_dir():
        sys.exit(f"missing dataset: {JANELA_10}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(), transforms.Normalize(base.MEAN, base.STD)])

    dist = base.load_split(DISTINTOS, "teste", "distintos")
    j15 = marcar_pos_corte(base.load_split(JANELA_15, "teste", "janela15"), JANELA_15)
    j10 = marcar_pos_corte(base.load_split(JANELA_10, "teste", "janela10"), JANELA_10)

    todos = dist + j15 + j10
    indice = {r["path"]: i for i, r in enumerate(todos)}
    print(f"device={device} | distintos={len(dist)} j15={len(j15)} j10={len(j10)} "
          f"unique={len(todos)}", flush=True)

    conjuntos = {
        "D distintos": dist,
        "J15 janela 0.5s": j15,
        "J15 pre-corte": [r for r in j15 if not r["pos_corte"]],
        "J15 pos-corte": [r for r in j15 if r["pos_corte"]],
        "J10 janela 10 quadros": j10,
        "J10 pre-corte": [r for r in j10 if not r["pos_corte"]],
        "J10 pos-corte": [r for r in j10 if r["pos_corte"]],
        "U15 distintos+J15": dist + j15,
        "U10 distintos+J10": dist + j10,
    }
    for nome, linhas in conjuntos.items():
        print(f"  {nome}: {len(linhas)} frames | "
              f"dev={dict(Counter(r['deviation'] for r in linhas))}", flush=True)

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    tamanhos = {k: len(v) for k, v in conjuntos.items()}
    resultados = {}
    if SAIDA.is_file():
        anterior = json.loads(SAIDA.read_text(encoding="utf-8"))
        if anterior.get("conjuntos") == tamanhos:
            resultados = anterior.get("resultados", {})
            print(f"resuming: {len(resultados)} models already scored", flush=True)
        else:
            print("test sets changed; recomputing every model", flush=True)

    for nome, caminho in base.CHECKPOINTS.items():
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
        model = base.TwoHead(backbone, dim)
        model.load_state_dict(ck["state_dict"])
        model.eval().to(device)

        t0 = time.time()
        pred_o = [0] * len(todos)
        pred_d = [0] * len(todos)
        loader = DataLoader(base.Frames(todos, transform), batch_size=64, shuffle=False, num_workers=0)
        with torch.inference_mode():
            for x, idx in loader:
                ol, dl = model(x.to(device))
                for k, i in enumerate(idx.tolist()):
                    pred_o[i] = int(ol[k].argmax())
                    pred_d[i] = int(dl[k].argmax())
        print(f"{nome}: predicted {len(todos)} frames in {time.time()-t0:.0f}s", flush=True)
        del model, backbone

        resultados[nome] = {}
        for rotulo, linhas in conjuntos.items():
            ids = [indice[r["path"]] for r in linhas]
            to = [base.OBSTACLE_CLASSES.index(r["obstacle"]) for r in linhas]
            td = [base.DEVIATION_CLASSES.index(r["deviation"]) for r in linhas]
            po = [pred_o[i] for i in ids]
            pd = [pred_d[i] for i in ids]
            joint = sum(1 for a, b, c, e in zip(to, td, po, pd) if a == c and b == e)
            resultados[nome][rotulo] = {
                "frames": len(linhas),
                "obstacle": base.metrics(to, po, base.OBSTACLE_CLASSES),
                "deviation": base.metrics(td, pd, base.DEVIATION_CLASSES),
                "joint_accuracy": round(joint / max(1, len(linhas)), 6)}
        SAIDA.write_text(json.dumps({"conjuntos": tamanhos, "resultados": resultados},
                                    indent=2, ensure_ascii=False), encoding="utf-8")

    for rotulo in conjuntos:
        print(f"\n=== {rotulo} ({len(conjuntos[rotulo])} frames) ===")
        print(f"{'model':26s} {'obs F1':>8s} {'dev F1':>8s} {'joint':>8s}")
        for nome, r in sorted(resultados.items(), key=lambda kv: -kv[1][rotulo]["joint_accuracy"]):
            v = r[rotulo]
            print(f"{nome:26s} {v['obstacle']['macro_f1_presentes']:8.4f} "
                  f"{v['deviation']['macro_f1_presentes']:8.4f} {v['joint_accuracy']:8.4f}")

    print(f"\nwrote {SAIDA}")


if __name__ == "__main__":
    main()
