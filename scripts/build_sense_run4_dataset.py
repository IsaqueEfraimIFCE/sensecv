"""Materialize the run4 MobileNetV1 dataset (33,118 / 6,172 / 3,970) on the
Desktop, in the file tree and label style of the Drive `sense` dataset.

Layout (mirrors sense/teste/):
    sense_run4_mobilenetv1_224/
        dataset_info.json
        <split>/images_x_y_224/<n>.jpg
        <split>/labels.txt          "<n>.jpg <obstacle_code> <deviation_code>"
        <split>/proveniencia.csv    sequential name -> original frame
"""
import csv, json, re, shutil, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

TMP = Path(r"C:\Users\Isaque\Desktop\sense_run4_experiments")
OUT = Path(r"C:\Users\Isaque\Desktop\sense_run4_mobilenetv1_224")
CACHE = OUT / "_cache_auto_trim"
IMG_DIRNAME = "images_x_y_224"
OBSTACLE_CODE = {"free_path": 0, "obstacle": 1}
DEVIATION_CODE = {"left": 0, "right": 1, "none": 2}

exec((TMP / "referencia" / "repro_split.py").read_text(encoding="utf-8").split("base = _pair_task_dirs")[0])

STATS = json.loads((TMP / "referencia" / "sibgrapi_run_stats" / "run4_1.txt").read_text(encoding="utf-8"))
br = STATS["background_reduction"]
VAL_CLIPS = set(br["validation"]["obstacle"]["clips"]) | set(br["validation"]["deviation"]["clips"])


def auto_trim_rows_materialized(existing):
    """Same eligibility rules as the trainer, but the frames are written out."""
    existing_clips = {r["clip"] for r in existing}
    existing_dates = {m.group(0) for c in existing_clips if (m := DATE_RE.search(c))}
    ledger = {}
    if (AUTO / "sources.csv").is_file():
        with (AUTO / "sources.csv").open(encoding="utf-8-sig", newline="") as f:
            ledger = {r.get("output_folder", ""): r for r in csv.DictReader(f)}
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = []
    for folder in sorted(p for p in AUTO.iterdir() if p.is_dir()):
        clip = folder.name
        date = DATE_RE.search(clip)
        if (clip in existing_clips or compound_caixa(clip)
                or collection_day(clip) in TEST_DAYS
                or (date and date.group(0) in existing_dates)):
            continue
        info = None
        try:
            info = json.loads((folder / "cut_info.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            info = ledger.get(clip)
        event = (info or {}).get("event_type", "").lower()
        side = (info or {}).get("side", "").upper()
        obstacle = "obstacle" if event in {"desvio", "parada"} else "free_path" if event == "livre" else None
        deviation = ({"LEFT": "left", "RIGHT": "right"}.get(side, "none") if event == "desvio"
                     else "none" if event in {"parada", "livre"} else None)
        video = next(iter(sorted(folder.glob("*.mp4"))), None)
        if not obstacle or not deviation or video is None:
            continue
        cap = cv2.VideoCapture(str(video))
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            target = CACHE / f"{clip}__f{index:05d}.jpg"
            if not target.exists():
                frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                if not cv2.imwrite(str(target), frame):
                    raise OSError(f"could not write {target}")
            rows.append({"path": str(target), "clip": clip, "obstacle": obstacle,
                         "deviation": deviation, "frame_index": index})
            index += 1
        cap.release()
    return rows


def balance_background(rows, label_name, background):
    groups, foreground = defaultdict(list), []
    for row in rows:
        (groups[row["clip"]] if row[label_name] == background else foreground).append(row)
    available = sum(map(len, groups.values()))
    target = min(available, len(foreground))
    if available <= target:
        return list(rows)
    quotas, fractions, allocated = {}, [], 0
    for clip, values in sorted(groups.items()):
        exact = len(values) * target / available
        quotas[clip] = int(np.floor(exact))
        allocated += quotas[clip]
        fractions.append((exact - quotas[clip], clip))
    for _, clip in sorted(fractions, key=lambda i: (-i[0], i[1]))[: target - allocated]:
        quotas[clip] += 1
    selected = set()
    for clip, values in groups.items():
        keep = quotas[clip]
        if keep:
            idx = np.floor((np.arange(keep) + .5) * len(values) / keep).astype(int)
            selected.update(id(values[i]) for i in idx)
    return [r for r in rows if r[label_name] != background or id(r) in selected]


def make_head_balanced_rows(rows):
    out = []
    for r in balance_background(rows, "obstacle", "free_path"):
        i = dict(r); i["head"] = "obstacle"; out.append(i)
    for r in balance_background(rows, "deviation", "none"):
        i = dict(r); i["head"] = "deviation"; out.append(i)
    return out


SWAP = {"left": "right", "right": "left", "none": "none"}


def augment(rows):
    """Verbatim augment() from the trainer: deterministic hflip duplication."""
    out = list(rows)
    for row in rows:
        flip = dict(row)
        flip["flip"] = True
        flip["deviation"] = SWAP[row["deviation"]]
        out.append(flip)
    return out


def emit(split, rows):
    d = OUT / split
    images = d / IMG_DIRNAME
    images.mkdir(parents=True, exist_ok=True)
    labels, prov = [], []
    for n, row in enumerate(rows, start=1):
        name = f"{n}.jpg"
        if row.get("flip"):
            # The trainer flips the decoded PIL image; mirror that, then re-encode
            # at the project's jpeg quality (95, as in the July-24 build_info).
            with Image.open(row["path"]) as im:
                im.convert("RGB").transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(
                    images / name, "JPEG", quality=95)
        else:
            shutil.copyfile(row["path"], images / name)
        labels.append(f"{name} {OBSTACLE_CODE[row['obstacle']]} {DEVIATION_CODE[row['deviation']]}")
        prov.append({"arquivo": name, "clip": row["clip"], "frame_index": row["frame_index"],
                     "cabeca": row.get("head", ""), "flip": int(bool(row.get("flip"))),
                     "obstacle_label": row["obstacle"],
                     "deviation_label": row["deviation"], "origem": row["path"]})
    (d / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    with (d / "proveniencia.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prov[0].keys()))
        w.writeheader()
        w.writerows(prov)
    return {"frames": len(rows), "clips": len({r["clip"] for r in rows}),
            "quadros_originais_distintos": len({r["path"] for r in rows}),
            "espelhados": sum(1 for r in rows if r.get("flip")),
            "obstacle": dict(Counter(r["obstacle"] for r in rows)),
            "deviation": dict(Counter(r["deviation"] for r in rows))}


start = time.time()
base = _pair_task_dirs(BASE)
july_all = _pair_task_dirs(JULY)
known = {r["clip"] for r in base}
rows = base + [r for r in july_all if r["clip"] not in known]
print(f"[{time.time()-start:6.1f}s] base+july rows: {len(rows)}", flush=True)
auto = auto_trim_rows_materialized(rows)
print(f"[{time.time()-start:6.1f}s] auto-trim frames decoded: {len(auto)}", flush=True)
allrows = rows + auto

test = [r for r in allrows if collection_day(r["clip"]) in set(TEST_DAYS)]
val = [r for r in allrows if r["clip"] in VAL_CLIPS]
train = [r for r in allrows if collection_day(r["clip"]) not in set(TEST_DAYS) and r["clip"] not in VAL_CLIPS]

summary = {}
summary["treino"] = emit("treino", make_head_balanced_rows(train))
print(f"[{time.time()-start:6.1f}s] treino written", flush=True)
summary["validacao"] = emit("validacao", make_head_balanced_rows(val))
print(f"[{time.time()-start:6.1f}s] validacao written", flush=True)
summary["teste"] = emit("teste", test)
print(f"[{time.time()-start:6.1f}s] teste written", flush=True)

info = {
    "format": "sensecv_allframes_dataset_v1",
    "variant": "run4_mobilenetv1_reduzido_por_cabeca",
    "images": sum(v["frames"] for v in summary.values()),
    "frames": sum(v["frames"] for v in summary.values()),
    "splits": {k: v["frames"] for k, v in summary.items()},
    "policy": (
        "same ingestion; test days fixed (22, 24); validation kept separate; "
        "per-head background reduction applied to treino and validacao "
        "(free_path -> obstacle count, none -> left+right count); teste natural; "
        "no sequences"
    ),
    "augmentacao": (
        "A unica augmentacao do treinador e o espelhamento horizontal deterministico "
        "de treino (augment(): cada linha e duplicada com left<->right trocados), "
        "que dobra o treino para train_effective = 66236. Ela NAO esta materializada "
        "aqui para preservar a contagem 33118; rode aplicar_flip.py para gerar "
        "treino_efetivo/ com as 66236 linhas. validacao e teste nunca sao aumentados."
    ),
    "preprocessamento": (
        "Fontes ja em 224x224 (recorte lateral/desvio aplicado a montante). "
        "O treinador aplica Resize((224,224), BICUBIC) + ToTensor + "
        "Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]); "
        "o Resize e no-op sobre estas imagens. Nenhum crop aleatorio e usado."
    ),
    "label_format": "frame_name obstacle_label deviation_label",
    "obstacle_labels": OBSTACLE_CODE,
    "deviation_labels": DEVIATION_CODE,
    "source_trainer": "scripts/mobilenetv1_two_head_all_frames.py",
    "reference_run": "kaggle anamariabarrosnonato/ss2ere sibgrapi/run4/final_statistics1.txt",
    "sources": {
        "base": "isaqueefraim/dataset8 kaggle_allframes_datasets_224",
        "july24": "isaqueefraim/dataset9 kaggle_24-07-2026_ufc_exports_unique_224",
        "auto_trim": "isaqueefraim/datasetandrefinal Todos_cortes_auto_trim/deviation",
    },
    "detalhe_por_split": summary,
    "nota_duplicatas": (
        "Uma imagem mantida pelas duas cabecas aparece duas vezes (uma linha por "
        "amostra de treino), reproduzindo train_reduced_per_head. "
        "proveniencia.csv traz a coluna 'cabeca' para colapsar em imagens unicas."
    ),
    "contagens_de_referencia": {
        "treino": {"frames": 33118, "free_path": 16051, "obstacle": 17067,
                   "left": 6992, "right": 7062, "none": 19064},
        "validacao": {"frames": 6172, "free_path": 3008, "obstacle": 3164,
                      "left": 1316, "right": 1444, "none": 3412},
        "teste": {"frames": 3970, "free_path": 1740, "obstacle": 2230,
                  "left": 1247, "right": 983, "none": 1740},
    },
}
(OUT / "dataset_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
shutil.copyfile(TMP / "aplicar_flip.py", OUT / "aplicar_flip.py")
shutil.rmtree(CACHE, ignore_errors=True)
print(json.dumps(summary, indent=2))
print(f"done in {time.time()-start:.1f}s -> {OUT}")
