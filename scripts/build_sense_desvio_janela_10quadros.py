"""C': left/right deviation clips, window extended 10 FRAMES past the cut's end.

Same construction as build_sense_desvio_janela.py, with one change: the tail is
counted in decoded frames instead of seconds. The window is [start, end] plus
the next EXTENSAO_QUADROS frames that follow the cut, so the tail length is
exact and independent of each recording's fps (0.5 s was ~15 frames at 30 fps).
All frames carry obstacle = obstacle and the clip's own deviation label.
"""
import csv, json, time
from collections import Counter, defaultdict
from pathlib import Path
import cv2

ROOT = Path(r"C:\Users\Isaque\sensecv")
SRC = Path(r"C:\Users\Isaque\Desktop\sense_run4_mobilenetv1_224")
OUT = Path(r"C:\Users\Isaque\Desktop\sense_run4_desvio_janela_10q_224")
AUTO = ROOT / "data/derived/manifest_exports/Todos_cortes_auto_trim/deviation"
UP = ROOT / "data/uploaded_datasets"
UFC = ROOT / "data/derived/kaggle_24-07-2026_ufc_exports_unique_224/provenance.csv"
HIST = ROOT / "data/history.json"
IMG = "images_x_y_224"
EXTENSAO_QUADROS = 10
TAMANHO = 224
OBSTACLE_CODE = {"free_path": 0, "obstacle": 1}
DEVIATION_CODE = {"left": 0, "right": 1, "none": 2}

# ---- which clips, and which split they belong to in run4 -------------------
clips = {}
for split in ("treino", "validacao", "teste"):
    for r in csv.DictReader((SRC / split / "proveniencia.csv").open(encoding="utf-8", newline="")):
        if r["deviation_label"] in ("left", "right"):
            clips.setdefault(r["clip"], {"split": split, "deviation": r["deviation_label"]})
print(f"left/right clips: {len(clips)}")

# ---- resolve every clip to (start, end, source video) ----------------------
export_to_clip = {}
for r in csv.DictReader(UFC.open(encoding="utf-8-sig", newline="")):
    export_to_clip[r["export_folder"]] = r["source_clip"]
historico = {}
for record in json.loads(HIST.read_text(encoding="utf-8")):
    clip = export_to_clip.get(record.get("folder", ""))
    if clip:
        historico[clip] = record

resolvidos, sem_fonte = {}, []
for clip, meta in clips.items():
    cut = AUTO / clip / "cut_info.json"
    info = None
    if cut.is_file():
        info = json.loads(cut.read_text(encoding="utf-8"))
    elif clip in historico:
        info = historico[clip]
    if not info:
        sem_fonte.append((clip, "sem metadados de corte"))
        continue
    video = UP.joinpath(*info["source_display" if "source_display" in info else "source_clip"].split("/")) / "video.mp4"
    if not video.is_file():
        sem_fonte.append((clip, f"sem video: {video}"))
        continue
    resolvidos[clip] = {**meta, "start": float(info["start"]), "end": float(info["end"]),
                        "video": video}
print(f"resolved: {len(resolvidos)} | unresolved: {len(sem_fonte)}")
for clip, motivo in sem_fonte:
    print("   -", clip, motivo)

# ---- decode each source video once, keeping cut + 10 following frames ------
por_video = defaultdict(list)
for clip, m in resolvidos.items():
    por_video[m["video"]].append(clip)

for split in ("treino", "validacao", "teste"):
    (OUT / split / IMG).mkdir(parents=True, exist_ok=True)
saida = {s: [] for s in ("treino", "validacao", "teste")}
truncados = []
start_time = time.time()

for i, (video, lista) in enumerate(sorted(por_video.items()), start=1):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # decode far enough that the 10-frame tail of the latest cut is reachable
    limite = max(resolvidos[c]["end"] for c in lista) + (EXTENSAO_QUADROS + 2) / fps
    janelas = [(c, resolvidos[c]["start"], resolvidos[c]["end"]) for c in lista]
    coletado = defaultdict(list)
    pos_corte_count = Counter()
    indice = 0
    while True:
        pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        ok, frame = cap.read()
        if not ok:
            break
        t = pos if pos > 0 or indice == 0 else indice / fps
        if t > limite + 1e-6:
            break
        redimensionado = None
        for clip, ini, fim in janelas:
            if t < ini - 1e-6:
                continue
            depois = t > fim + 1e-6
            if depois and pos_corte_count[clip] >= EXTENSAO_QUADROS:
                continue
            if redimensionado is None:
                redimensionado = cv2.resize(frame, (TAMANHO, TAMANHO), interpolation=cv2.INTER_AREA)
            coletado[clip].append((t, depois, redimensionado))
            if depois:
                pos_corte_count[clip] += 1
        indice += 1
    duracao = indice / fps if fps else 0.0
    cap.release()
    for clip, ini, fim in janelas:
        if pos_corte_count[clip] < EXTENSAO_QUADROS:
            truncados.append({"clip": clip, "quadros_pos_corte": pos_corte_count[clip],
                              "desejado": EXTENSAO_QUADROS, "duracao_video": round(duracao, 3)})
        m = resolvidos[clip]
        janela_fim = max((t for t, _, _ in coletado[clip]), default=fim)
        for ordem, (t, depois, frame) in enumerate(coletado[clip]):
            saida[m["split"]].append({"clip": clip, "deviation": m["deviation"], "tempo_s": round(t, 4),
                                      "ordem": ordem, "frame": frame, "pos_corte": int(depois),
                                      "janela_inicio": round(ini, 3), "janela_fim": round(janela_fim, 3),
                                      "corte_fim": round(m["end"], 3), "video": str(video)})
    if i % 40 == 0:
        print(f"[{time.time()-start_time:6.1f}s] {i}/{len(por_video)} videos", flush=True)

# ---- write in the distinct-source-frames style -----------------------------
resumo = {}
for split, linhas in saida.items():
    d = OUT / split
    labels, prov = [], []
    for n, row in enumerate(linhas, start=1):
        nome = f"{n}.jpg"
        if not cv2.imwrite(str(d / IMG / nome), row["frame"]):
            raise OSError(f"could not write {d / IMG / nome}")
        labels.append(f"{nome} {OBSTACLE_CODE['obstacle']} {DEVIATION_CODE[row['deviation']]}")
        prov.append({"arquivo": nome, "clip": row["clip"], "ordem_na_janela": row["ordem"],
                     "tempo_s": row["tempo_s"], "janela_inicio": row["janela_inicio"],
                     "janela_fim": row["janela_fim"], "corte_fim_original": row["corte_fim"],
                     "pos_corte": row["pos_corte"],
                     "obstacle_label": "obstacle", "deviation_label": row["deviation"],
                     "origem": row["video"]})
    (d / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    with (d / "proveniencia.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prov[0].keys()))
        w.writeheader()
        w.writerows(prov)
    resumo[split] = {"frames": len(prov), "clips": len({p["clip"] for p in prov}),
                     "quadros_pos_corte": sum(p["pos_corte"] for p in prov),
                     "deviation": dict(Counter(p["deviation_label"] for p in prov)),
                     "obstacle": {"obstacle": len(prov)}}

info = {
    "format": "sensecv_allframes_dataset_v1",
    "variant": "run4_desvio_janela_estendida_10_quadros",
    "images": sum(v["frames"] for v in resumo.values()),
    "frames": sum(v["frames"] for v in resumo.values()),
    "splits": {k: v["frames"] for k, v in resumo.items()},
    "policy": (
        "Apenas clipes de desvio left/right do conjunto run4. A janela de cada "
        "clipe e [start, end] mais os 10 quadros seguintes do video original, "
        "contados em quadros decodificados e nao em segundos. Todos os quadros "
        "recebem obstacle_label = obstacle e o deviation_label do proprio clipe. "
        "A divisao treino/validacao/teste e a mesma do run4."
    ),
    "extensao_quadros": EXTENSAO_QUADROS,
    "label_format": "frame_name obstacle_label deviation_label",
    "obstacle_labels": OBSTACLE_CODE,
    "deviation_labels": DEVIATION_CODE,
    "fontes": {
        "cortes": "data/derived/manifest_exports/Todos_cortes_auto_trim/deviation/<clip>/cut_info.json",
        "exports_24_07": "data/history.json + kaggle_24-07-2026_ufc_exports_unique_224/provenance.csv",
        "videos": "data/uploaded_datasets/<sessao>/<pasta>/<NN>/video.mp4",
    },
    "clipes_resolvidos": len(resolvidos),
    "clipes_nao_resolvidos": [{"clip": c, "motivo": m} for c, m in sem_fonte],
    "janelas_com_cauda_incompleta": truncados,
    "nota": (
        "Os quadros sao redecodificados do video original, entao os quadros do "
        "trecho do corte nao sao byte-identicos aos do dataset run4. "
        "proveniencia.csv traz 'pos_corte' = 1 para os quadros da extensao; "
        "sao no maximo 10 por clipe."
    ),
    "detalhe_por_split": resumo,
}
(OUT / "dataset_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(resumo, indent=2, ensure_ascii=False))
print(f"windows with a short tail: {len(truncados)}")
print(f"done in {time.time()-start_time:.1f}s -> {OUT}")
