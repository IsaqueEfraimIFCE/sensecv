# -*- coding: utf-8 -*-
"""Train a two-head MobileNet (128x128, alpha=0.25) on the SenseCV Kaggle dataset.

Designed to run on Kaggle (TensorFlow available there). Defaults point at the
uploaded Kaggle dataset:

  --labels      /kaggle/input/datasets/isaqueefraim/dataset7/labels.txt
  --images-dir  /kaggle/input/datasets/isaqueefraim/dataset7/dataset/dataset

labels.txt has three columns ("file_name obstacle_class deviation_class"); the
collection day is parsed straight from each file name (it contains the group, e.g.
SenseCV-06-06-2026-IFCE__...__frame_00001.jpg). Images are resolved by basename in
--images-dir (the flat folder that holds the JPGs).

Split is **by collection day** to avoid temporal bias / leakage (frames from the
same clip or day never appear in two splits):
  test  = images from --test-day only   (default 2026-06-20)
  val   = images from --val-days         (default: latest non-test day that has
          BOTH deviation directions and is not the only source of no-obstacle
          frames, so the deviation head is actually measured)
  train = every remaining day            (disjoint from val and test; keeps the
          no-obstacle days so the obstacle head can learn negatives)

Images are decoded as RGB (3 channels), resized to --img-size (default 128) and
fed to MobileNet preprocess_input. Two output heads:
  obstacle  -> 1 unit, sigmoid          (0 = no obstacle, 1 = obstacle)
  deviation -> 3 units, softmax          (0 = left, 1 = right, 2 = none)

Class imbalance (the dataset is mostly obstacle) is handled with per-head sample
weights from inverse class frequency (computed on the train split). Training is
two-phase: a frozen-base warmup then a low-LR fine-tune of the whole network. The
best model (by val loss) is saved to --out and finally evaluated on the test day.

  python train_two_head_mobilenet.py \
      --img-size 128 --alpha 0.25 --epochs 40 --batch 32 \
      --test-day 2026-06-20 \
      --out /kaggle/working/two_head_mobilenet_128_a025.keras
"""
import argparse
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE
_DATE_RE = re.compile(r"(\d{2})-(\d{2})-(\d{4})")


def day_of(source_group):
    """Collection date parsed from a group name like SenseCV-20-06-2026-BECE."""
    m = _DATE_RE.search(source_group or "")
    if not m:
        return None
    d, mo, y = (int(x) for x in m.groups())
    try:
        return date(y, mo, d)
    except ValueError:
        return None


def read_labels(labels_path, images_dir):
    """Read labels.txt -> (files, obstacle, deviation, days).

    labels.txt rows are "file_name obstacle_class deviation_class". Images are
    resolved by basename inside images_dir (a flat folder); the collection day is
    parsed from the file name (which carries the group, e.g. SenseCV-06-06-2026).
    """
    files, obst, devs, days = [], [], [], []
    missing = 0
    with labels_path.open(encoding="utf-8") as f:
        first = f.readline().split()
        # Skip a header row if present; otherwise treat it as data.
        rows = []
        if not (len(first) == 3 and first[1].isdigit() and first[2].isdigit()):
            pass  # header consumed
        else:
            rows.append(first)
        rows.extend(line.split() for line in f)
        for parts in rows:
            if len(parts) != 3:
                continue
            rel, o, d = parts
            name = Path(rel).name
            p = images_dir / name
            if not p.is_file():
                p = images_dir / rel            # fallback: keep the relative prefix
            if not p.is_file():
                missing += 1
                continue
            files.append(str(p))
            obst.append(int(o)); devs.append(int(d)); days.append(day_of(rel))
    if missing:
        print(f"[aviso] {missing} arquivos de imagem nao encontrados em {images_dir}")
    return files, np.array(obst, np.int64), np.array(devs, np.int64), np.array(days, dtype=object)


def inverse_freq_weights(labels):
    counts = Counter(int(x) for x in labels)
    total = len(labels); k = len(counts)
    return {c: total / (k * n) for c, n in counts.items()}


def make_dataset(files, obst, devs, w_o, w_d, img_size, batch, training, seed):
    paths = tf.constant(files)
    ds = tf.data.Dataset.from_tensor_slices(
        (paths, obst, devs,
         np.array([w_o[int(o)] for o in obst], np.float32),
         np.array([w_d[int(d)] for d in devs], np.float32)))
    if training:
        ds = ds.shuffle(min(len(files), 2048), seed=seed, reshuffle_each_iteration=True)

    def load(path, o, d, so, sd):
        img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)  # RGB
        img = tf.image.resize(img, (img_size, img_size))
        if training:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.1)
        img = tf.keras.applications.mobilenet.preprocess_input(tf.cast(img, tf.float32))
        return (img,
                {"obstacle": tf.cast(o, tf.float32), "deviation": d},
                {"obstacle": so, "deviation": sd})

    return ds.map(load, num_parallel_calls=AUTOTUNE).batch(batch).prefetch(AUTOTUNE)


def build_model(img_size, alpha):
    try:
        base = tf.keras.applications.MobileNet(
            input_shape=(img_size, img_size, 3), alpha=alpha,
            include_top=False, weights="imagenet", pooling="avg")
    except Exception as e:
        print(f"[aviso] pesos imagenet indisponiveis ({e}); treinando do zero")
        base = tf.keras.applications.MobileNet(
            input_shape=(img_size, img_size, 3), alpha=alpha,
            include_top=False, weights=None, pooling="avg")
    x = tf.keras.layers.Dropout(0.2)(base.output)
    obstacle = tf.keras.layers.Dense(1, activation="sigmoid", name="obstacle")(x)
    deviation = tf.keras.layers.Dense(3, activation="softmax", name="deviation")(x)
    # Dict outputs (not a list) so dict losses/metrics/targets resolve by name on
    # Keras 3 (Kaggle); a list output + dict loss raises KeyError: 0 there.
    model = tf.keras.Model(base.input, {"obstacle": obstacle, "deviation": deviation})
    return model, base


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss={"obstacle": "binary_crossentropy",
              "deviation": "sparse_categorical_crossentropy"},
        metrics={"obstacle": ["accuracy"], "deviation": ["accuracy"]},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels",
                    default="/kaggle/input/datasets/isaqueefraim/dataset7/labels.txt")
    ap.add_argument("--images-dir",
                    default="/kaggle/input/datasets/isaqueefraim/dataset7/dataset/dataset")
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--warmup-epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--test-day", default="2026-06-20",
                    help="dia usado SO como teste (YYYY-MM-DD)")
    ap.add_argument("--val-days", default="",
                    help="dia(s) de validacao separados por virgula; "
                         "vazio = usa o ultimo dia que nao seja teste")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="/kaggle/working/two_head_mobilenet_128_a025.keras")
    # parse_known_args (not parse_args) so a notebook kernel's injected args
    # (e.g. "-f /root/.../kernel.json" in Colab/Jupyter) are ignored.
    args, _ = ap.parse_known_args()

    labels_path = Path(args.labels)
    images_dir = Path(args.images_dir)
    files, obst, devs, days = read_labels(labels_path, images_dir)
    if not files:
        raise SystemExit(f"Nenhuma imagem resolvida de {labels_path} em {images_dir}")
    files = np.array(files, dtype=object)

    test_day = date.fromisoformat(args.test_day)
    all_days = sorted({d for d in days if d is not None})
    non_test = [d for d in all_days if d != test_day]

    # Per-day class presence, to pick a non-degenerate default split.
    has_neg = {d: bool((obst[days == d] == 0).any()) for d in non_test}   # has no-obstacle frames
    has_left = {d: bool((devs[days == d] == 0).any()) for d in non_test}
    has_right = {d: bool((devs[days == d] == 1).any()) for d in non_test}

    if args.val_days.strip():
        val_days = {date.fromisoformat(s.strip()) for s in args.val_days.split(",") if s.strip()}
    else:
        # Keep no-obstacle days in train (they are the only source of obstacle=0
        # negatives); pick the latest remaining day that has BOTH deviation
        # directions so the deviation head is actually validated.
        cand = [d for d in non_test if has_left[d] and has_right[d] and not has_neg[d]]
        val_days = {cand[-1]} if cand else ({non_test[-1]} if non_test else set())
    val_days = {d for d in val_days if d != test_day}
    train_days = [d for d in non_test if d not in val_days]

    print(f"{len(files)} imagens | dias: {[d.isoformat() for d in all_days]}")
    print(f"  treino = {[d.isoformat() for d in train_days]}")
    print(f"  val    = {[d.isoformat() for d in sorted(val_days)]}")
    print(f"  teste  = [{test_day.isoformat()}]")
    if not train_days or not val_days:
        raise SystemExit("Split invalido: treino ou validacao vazio. Ajuste --val-days/--test-day.")
    overlap = (set(train_days) & val_days) | (set(train_days) & {test_day}) | (val_days & {test_day})
    if overlap:
        raise SystemExit(f"Vazamento temporal: dias compartilhados entre splits: {overlap}")

    day_arr = days
    tr_i = np.array([i for i in range(len(files)) if day_arr[i] in set(train_days)])
    val_i = np.array([i for i in range(len(files)) if day_arr[i] in val_days])
    test_i = np.array([i for i in range(len(files)) if day_arr[i] == test_day])
    for name, ii in (("treino", tr_i), ("val", val_i), ("teste", test_i)):
        oc, dc = Counter(obst[ii].tolist()), Counter(devs[ii].tolist())
        print(f"  {name}: {len(ii)} imgs | obstacle {dict(oc)} | deviation {dict(dc)}")
        if len(ii) == 0:
            raise SystemExit(f"Split '{name}' vazio. Ajuste --val-days/--test-day.")
        if len(oc) < 2:
            print(f"    [aviso] '{name}' tem so uma classe de obstaculo "
                  f"-> a cabeca obstacle nao e medida/treinada aqui.")
        if {0, 1} - set(dc):
            print(f"    [aviso] '{name}' nao cobre as duas direcoes de desvio (left/right).")

    w_o = inverse_freq_weights(obst[tr_i])
    w_d = inverse_freq_weights(devs[tr_i])
    print("sample weights obstacle:", {k: round(v, 2) for k, v in w_o.items()},
          "deviation:", {k: round(v, 2) for k, v in w_d.items()})

    def subset(i):
        return [str(x) for x in files[i]], obst[i], devs[i]
    tf_files, tf_o, tf_d = subset(tr_i)
    vf_files, vf_o, vf_d = subset(val_i)
    xf_files, xf_o, xf_d = subset(test_i)
    train_ds = make_dataset(tf_files, tf_o, tf_d, w_o, w_d, args.img_size, args.batch, True, args.seed)
    val_ds = make_dataset(vf_files, vf_o, vf_d, w_o, w_d, args.img_size, args.batch, False, args.seed)
    test_ds = make_dataset(xf_files, xf_o, xf_d, w_o, w_d, args.img_size, args.batch, False, args.seed)

    model, base = build_model(args.img_size, args.alpha)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = tf.keras.callbacks.ModelCheckpoint(str(out), monitor="val_loss",
                                              save_best_only=True, verbose=1)
    early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                             restore_best_weights=True)

    # Phase 1: frozen base warmup.
    base.trainable = False
    compile_model(model, 1e-3)
    print("\n== Fase 1: warmup (base congelada) ==")
    h1 = model.fit(train_ds, validation_data=val_ds,
                   epochs=args.warmup_epochs, callbacks=[ckpt, early])

    # Phase 2: fine-tune the whole network at a low LR.
    base.trainable = True
    compile_model(model, 1e-4)
    print("\n== Fase 2: fine-tuning (rede inteira, LR baixo) ==")
    h2 = model.fit(train_ds, validation_data=val_ds, initial_epoch=len(h1.epoch),
                   epochs=args.epochs, callbacks=[ckpt, early])

    model.save(out)
    hist = {k: [float(x) for x in v] for k, v in {**h1.history,
            **{k: h1.history.get(k, []) + v for k, v in h2.history.items()}}.items()}
    (out.with_suffix(".history.json")).write_text(json.dumps(hist, indent=2))

    # Final, unbiased evaluation on the held-out test day.
    print(f"\n== Avaliacao no dia de teste ({test_day.isoformat()}) ==")
    test_metrics = model.evaluate(test_ds, return_dict=True, verbose=1)
    print(json.dumps({k: round(float(v), 4) for k, v in test_metrics.items()}, indent=2))
    (out.with_suffix(".test_metrics.json")).write_text(json.dumps({
        "test_day": test_day.isoformat(),
        "val_days": [d.isoformat() for d in sorted(val_days)],
        "train_days": [d.isoformat() for d in train_days],
        "metrics": {k: float(v) for k, v in test_metrics.items()},
    }, indent=2))
    print(f"\nModelo salvo em {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
