"""Append the 3 July-24 clips whose cut timing is absent from the pruned
history.json to the 10-frame janela dataset.

Same frame-matching recovery as build_sense_desvio_janela_recuperar.py — the
export's first frame is located in the original recording — but the tail is the
10 frames following the cut instead of 0.5 s, matching
build_sense_desvio_janela_10quadros.py. All three clips are in the test split.
"""
import csv, json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(r"C:\Users\Isaque\sensecv")
OUT = Path(r"C:\Users\Isaque\Desktop\sense_run4_desvio_janela_10q_224")
RUN4 = Path(r"C:\Users\Isaque\Desktop\sense_run4_mobilenetv1_224")
UP = ROOT / "data/uploaded_datasets/SenseCV-24-07-2026-UFC/SenseCV-24-07-2026-UFC"
EXPORTS = ROOT / "data/exports"
PROV = ROOT / "data/derived/kaggle_24-07-2026_ufc_exports_unique_224/provenance.csv"
IMG = "images_x_y_224"
EXTENSAO_QUADROS = 10
DEVIATION_CODE = {"left": 0, "right": 1, "none": 2}
ALVOS = ["SenseCV-24-07-2026-UFC_SenseCV-24-07-2026-UFC_21",
         "SenseCV-24-07-2026-UFC_SenseCV-24-07-2026-UFC_26",
         "SenseCV-24-07-2026-UFC_SenseCV-24-07-2026-UFC_30"]


def assinatura(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)


lados = {}
for r in csv.DictReader((RUN4 / "teste" / "proveniencia.csv").open(encoding="utf-8", newline="")):
    if r["clip"] in ALVOS:
        lados[r["clip"]] = r["deviation_label"]
prov = {r["source_clip"]: r for r in csv.DictReader(PROV.open(encoding="utf-8-sig", newline=""))}

destino = OUT / "teste"
labels = [l for l in (destino / "labels.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
registros = list(csv.DictReader((destino / "proveniencia.csv").open(encoding="utf-8", newline="")))
ja_presentes = {r["clip"] for r in registros}
proximo = len(labels) + 1
adicionados = Counter()
recuperados = []

for clip in ALVOS:
    if clip in ja_presentes:
        print(f"{clip}: ja esta no dataset, ignorado")
        continue
    r = prov[clip]
    export_mp4 = next(iter(sorted((EXPORTS / r["export_folder"]).glob("*.mp4"))), None)
    fonte = UP / r["source_id"] / "video.mp4"
    if export_mp4 is None or not fonte.is_file():
        print(f"{clip}: sem mp4/fonte, ignorado")
        continue

    cap = cv2.VideoCapture(str(export_mp4))
    ok, primeiro = cap.read()
    cap.release()
    alvo = assinatura(primeiro)

    cap = cv2.VideoCapture(str(fonte))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    quadros, i = [], 0
    while True:
        pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        ok, frame = cap.read()
        if not ok:
            break
        quadros.append((pos if pos > 0 or i == 0 else i / fps, frame))
        i += 1
    cap.release()

    custo, inicio = min((float(np.abs(assinatura(f) - alvo).mean()), t) for t, f in quadros)
    n = int(r["frames"])
    fim = inicio + n / fps

    # the cut itself, then at most EXTENSAO_QUADROS frames after it
    selecionados, depois_count = [], 0
    for t, frame in quadros:
        if t < inicio - 1e-6:
            continue
        depois = t > fim + 1e-6
        if depois:
            if depois_count >= EXTENSAO_QUADROS:
                break
            depois_count += 1
        selecionados.append((t, depois, frame))

    janela_fim = max((t for t, _, _ in selecionados), default=fim)
    recuperados.append({"clip": clip, "export": r["export_folder"], "start": round(inicio, 3),
                        "end": round(fim, 3), "custo_casamento": round(custo, 3),
                        "quadros_pos_corte": depois_count,
                        "metodo": "casamento visual do primeiro quadro do export"})
    lado = lados[clip]
    for ordem, (t, depois, frame) in enumerate(selecionados):
        nome = f"{proximo}.jpg"
        imagem = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(str(destino / IMG / nome), imagem):
            raise OSError(nome)
        labels.append(f"{nome} 1 {DEVIATION_CODE[lado]}")
        registros.append({"arquivo": nome, "clip": clip, "ordem_na_janela": ordem,
                          "tempo_s": round(t, 4), "janela_inicio": round(inicio, 3),
                          "janela_fim": round(janela_fim, 3), "corte_fim_original": round(fim, 3),
                          "pos_corte": int(depois), "obstacle_label": "obstacle",
                          "deviation_label": lado, "origem": str(fonte)})
        proximo += 1
        adicionados[clip] += 1
    print(f"{clip}: start={inicio:.3f} end={fim:.3f} custo={custo:.2f} "
          f"quadros={adicionados[clip]} (pos_corte={depois_count})")

(destino / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
with (destino / "proveniencia.csv").open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(registros[0].keys()))
    w.writeheader()
    w.writerows(registros)

info = json.loads((OUT / "dataset_info.json").read_text(encoding="utf-8"))
resumo = info["detalhe_por_split"]["teste"]
resumo["frames"] = len(registros)
resumo["clips"] = len({r["clip"] for r in registros})
resumo["quadros_pos_corte"] = sum(int(r["pos_corte"]) for r in registros)
resumo["deviation"] = dict(Counter(r["deviation_label"] for r in registros))
resumo["obstacle"] = {"obstacle": len(registros)}
info["splits"]["teste"] = len(registros)
info["images"] = info["frames"] = sum(info["splits"].values())
info["clipes_resolvidos"] = info["clipes_resolvidos"] + len(recuperados)
info["clipes_nao_resolvidos"] = [c for c in info["clipes_nao_resolvidos"] if c["clip"] not in adicionados]
info["clipes_recuperados_por_casamento"] = recuperados
info["nota_casamento"] = (
    "history.json local esta podado e nao tem os exports 23/28/31. O inicio "
    "desses 3 clipes foi recuperado casando o primeiro quadro do export com o "
    "video original (erro medido em clipes com referencia: -0.010 a -0.024 s, "
    "menos de um quadro a 30 fps); o fim e start + n_quadros/fps."
)
(OUT / "dataset_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
print("teste agora:", resumo)
