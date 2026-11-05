"""Recover the exact [start, end] of an export by matching its own frames
against the original recording. Validated against data/history.json."""
import csv, json, sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(r"C:\Users\Isaque\sensecv")
UP = ROOT / "data/uploaded_datasets/SenseCV-24-07-2026-UFC/SenseCV-24-07-2026-UFC"
EXPORTS = ROOT / "data/exports"
PROV = ROOT / "data/derived/kaggle_24-07-2026_ufc_exports_unique_224/provenance.csv"
HIST = ROOT / "data/history.json"


def assinatura(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)


def frames_de(video, limite=None):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out, i = [], 0
    while True:
        pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        ok, frame = cap.read()
        if not ok or (limite and i >= limite):
            break
        out.append((pos if pos > 0 or i == 0 else i / fps, assinatura(frame)))
        i += 1
    cap.release()
    return out, fps


def casar(export_mp4, fonte_mp4):
    exp, _ = frames_de(export_mp4)
    src, fps = frames_de(fonte_mp4)
    primeira, ultima = exp[0][1], exp[-1][1]
    def melhor(alvo):
        custos = [(float(np.abs(s - alvo).mean()), t) for t, s in src]
        return min(custos)
    c0, t0 = melhor(primeira)
    c1, t1 = melhor(ultima)
    return t0, t1, c0, c1, len(exp), fps


prov = {r["export_folder"]: r for r in csv.DictReader(PROV.open(encoding="utf-8-sig", newline=""))}
hist = {rec["folder"]: rec for rec in json.loads(HIST.read_text(encoding="utf-8"))}

alvo = sys.argv[1:] or ["01_obstaculo_direita_desvio_esquerda", "02_obstaculo_direita_desvio_esquerda",
                        "04_obstaculo_direita_desvio_esquerda"]
for folder in alvo:
    r = prov[folder]
    pasta = EXPORTS / folder
    mp4 = next(iter(sorted(pasta.glob("*.mp4"))), None)
    fonte = UP / r["source_id"] / "video.mp4"
    if mp4 is None or not fonte.is_file():
        print(f"{folder}: sem mp4 ou fonte")
        continue
    t0, t1, c0, c1, n, fps = casar(mp4, fonte)
    rec = hist.get(folder)
    ref = f" | history start={rec['start']:.3f} end={rec['end']:.3f}" if rec else " | sem history"
    print(f"{folder[:34]:34s} src={r['source_id']} n={n:3d} fps={fps:.2f} "
          f"match start={t0:.3f} end={t1:.3f} custo={c0:.2f}/{c1:.2f}{ref}")
    if rec:
        print(f"{'':34s} erro start={t0-rec['start']:+.3f}s end={t1-rec['end']:+.3f}s")
