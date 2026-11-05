"""Build the two distinct-frame datasets from the already-materialized run4 set.

A: sense_run4_quadros_distintos_224   - each source frame once, both labels
B: sense_run4_quadros_distintos_aug_224 - A plus the trainer's deterministic
   horizontal flip on treino only (left<->right swapped), as in augment().
"""
import csv, json, shutil, time
from collections import Counter
from pathlib import Path
from PIL import Image

SRC = Path(r"C:\Users\Isaque\Desktop\sense_run4_mobilenetv1_224")
A = Path(r"C:\Users\Isaque\Desktop\sense_run4_quadros_distintos_224")
B = Path(r"C:\Users\Isaque\Desktop\sense_run4_quadros_distintos_aug_224")
IMG = "images_x_y_224"
OBSTACLE_CODE = {"free_path": 0, "obstacle": 1}
DEVIATION_CODE = {"left": 0, "right": 1, "none": 2}
SWAP = {"left": "right", "right": "left", "none": "none"}
SPLITS = ["treino", "validacao", "teste"]


def distinct_rows(split):
    """First occurrence of each source frame, in the run4 row order."""
    rows = list(csv.DictReader((SRC / split / "proveniencia.csv").open(encoding="utf-8", newline="")))
    seen, out = set(), []
    for r in rows:
        if r["origem"] in seen:
            continue
        seen.add(r["origem"])
        r = dict(r)
        r["cabecas"] = ""
        out.append(r)
    heads = {}
    for r in rows:
        heads.setdefault(r["origem"], set()).add(r["cabeca"])
    for r in out:
        r["cabecas"] = "+".join(sorted(heads[r["origem"]]))
    return out


def write(out_root, split, entries):
    d = out_root / split
    (d / IMG).mkdir(parents=True, exist_ok=True)
    labels, prov = [], []
    for n, (row, flip) in enumerate(entries, start=1):
        name = f"{n}.jpg"
        source = SRC / split / IMG / row["arquivo"]
        deviation = SWAP[row["deviation_label"]] if flip else row["deviation_label"]
        if flip:
            with Image.open(source) as im:
                im.convert("RGB").transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(
                    d / IMG / name, "JPEG", quality=95)
        else:
            shutil.copyfile(source, d / IMG / name)
        labels.append(f"{name} {OBSTACLE_CODE[row['obstacle_label']]} {DEVIATION_CODE[deviation]}")
        prov.append({"arquivo": name, "clip": row["clip"], "frame_index": row["frame_index"],
                     "cabecas": row["cabecas"], "flip": int(flip),
                     "obstacle_label": row["obstacle_label"], "deviation_label": deviation,
                     "origem": row["origem"]})
    (d / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    with (d / "proveniencia.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prov[0].keys()))
        w.writeheader()
        w.writerows(prov)
    return {"frames": len(prov), "clips": len({p["clip"] for p in prov}),
            "espelhados": sum(p["flip"] for p in prov),
            "obstacle": dict(Counter(p["obstacle_label"] for p in prov)),
            "deviation": dict(Counter(p["deviation_label"] for p in prov))}


def info(out_root, resumo, augmented):
    return {
        "format": "sensecv_allframes_dataset_v1",
        "variant": ("run4_quadros_distintos_com_flip" if augmented else "run4_quadros_distintos"),
        "images": sum(v["frames"] for v in resumo.values()),
        "frames": sum(v["frames"] for v in resumo.values()),
        "splits": {k: v["frames"] for k, v in resumo.items()},
        "policy": (
            "quadros distintos do conjunto run4 reduzido por cabeca (cada quadro "
            "uma unica vez, com os dois rotulos); mesma ingestao, mesmos dias de "
            "teste (22, 24), mesma divisao por clipe do run4"
            + ("; augmentacao de espelhamento horizontal aplicada apenas a treino "
               "(left<->right trocados), como em augment(); validacao e teste nunca "
               "sao aumentados" if augmented else "; sem augmentacao")
        ),
        "label_format": "frame_name obstacle_label deviation_label",
        "obstacle_labels": OBSTACLE_CODE,
        "deviation_labels": DEVIATION_CODE,
        "origem": "C:/Users/Isaque/Desktop/sense_run4_mobilenetv1_224",
        "nota": (
            "A coluna 'cabecas' em proveniencia.csv indica quais cabecas usaram o "
            "quadro no run4 (obstacle, deviation ou ambas); no run4 os quadros de "
            "ambas as cabecas aparecem duas vezes, aqui aparecem uma vez."
        ),
        "detalhe_por_split": resumo,
    }


start = time.time()
for root, augmented in ((A, False), (B, True)):
    resumo = {}
    for split in SPLITS:
        rows = distinct_rows(split)
        entries = [(r, False) for r in rows]
        if augmented and split == "treino":
            entries += [(r, True) for r in rows]
        resumo[split] = write(root, split, entries)
        print(f"[{time.time()-start:6.1f}s] {root.name}/{split}: {resumo[split]['frames']}", flush=True)
    (root / "dataset_info.json").write_text(
        json.dumps(info(root, resumo, augmented), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resumo, indent=2))
print(f"done in {time.time()-start:.1f}s")
