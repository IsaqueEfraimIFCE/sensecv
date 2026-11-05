---
type: entity
tags: [dataset, training, run4, sense, kaggle]
code_refs: [scripts/mobilenetv4_two_head_all_frames.py, scripts/convlstm_mobilenetv1_all_frames.py]
updated: 2026-08-16
---

# Sense-style datasets (run4 reproduction)

Four datasets built on 2026-08-16 outside the repo, in the file tree and label
style of the Drive `sense` dataset (`sensecv_allframes_dataset_v1`). All live
under `C:\Users\Isaque\Desktop\`, all at 224x224.

Shared layout, one folder per split:

```
<dataset>/
  dataset_info.json
  <treino|validacao|teste>/
    images_x_y_224/1.jpg .. N.jpg      # sequential, flat
    labels.txt                         # "<n>.jpg <obstacle_code> <deviation_code>"
    proveniencia.csv                   # sequential name -> clip, frame, source path
```

`obstacle_code`: `free_path` 0, `obstacle` 1. `deviation_code`: `left` 0,
`right` 1, `none` 2 — the same codes as `sense/dataset_info.json`.
`proveniencia.csv` is an addition: the sequential renaming would otherwise
destroy provenance.

## The four datasets

| folder | treino | validacao | teste | what it is |
|---|---|---|---|---|
| `sense_run4_mobilenetv1_224` | 33,118 | 6,172 | 3,970 | bit-exact reproduction of run4's MobileNetV1 split |
| `sense_run4_quadros_distintos_224` | 24,347 | 4,288 | 3,970 | each source frame once, both labels |
| `sense_run4_quadros_distintos_aug_224` | 48,694 | 4,288 | 3,970 | the above + mirrored training copies |
| `sense_run4_desvio_janela_1_5s_224` | 10,047 | 1,930 | 2,706 | left/right clips, window extended 0.5 s past the cut |

### run4 reproduction

Reproduces `anamariabarrosnonato/ss2ere` → `sibgrapi/run4/final_statistics1.txt`
exactly: treino 33,118 (free 16,051 / obstacle 17,067; left 6,992 / right 7,062
/ none 19,064), validacao 6,172, teste 3,970. The validation clip set was taken
from that artifact's `background_reduction` clip lists rather than re-drawn, so
the partition is the run's own. See [[allframes-split-provenance]] for the
derivation.

`aplicar_flip.py` ships inside the folder and regenerates `train_effective`
(66,236 rows) on demand.

### Distinct frames

run4's 33,118 training rows cover only **24,347 distinct frames**: 8,771 appear
twice because [[per-head-background-reduction]] emits one row per head. Here
each frame appears once carrying both labels. `proveniencia.csv` gains a
`cabecas` column (`obstacle`, `deviation`, or `obstacle+deviation`).

### Extended deviation windows

For every clip whose deviation label is `left`/`right`, the window
`[start, end + 0.5 s]` is decoded from the **original recording** — a 1.0 s cut
becomes 1.5 s — and every frame is kept, labelled `obstacle` + the clip's own
side. 327 of 328 clips resolved:

- 301 from `Todos_cortes_auto_trim/deviation/<clip>/cut_info.json`
- 24 from `data/history.json` + the July-24 package's `provenance.csv`
- 3 (exports 23/28/31, absent from the pruned local `history.json`) recovered by
  matching the export's first frame against its source recording; validated on
  clips with known timing to −0.010..−0.024 s, under one frame at 30 fps
- 1 unresolved: `SenseCV-02-06-2026-IFCE...Gimbal_01` has no `video.mp4`

`proveniencia.csv` flags extension frames with `pos_corte=1` (899 of the 2,706
test frames). These frames are re-decoded from source, so the cut portion is
not byte-identical to the same frames in the other datasets.

## Kaggle

Uploaded as the private dataset `anamariabarrosnonato/sensecv-run4-sense`
(919 MB, both tars; Kaggle auto-extracts them into a nested path).

## Related

- [[allframes-split-provenance]] — where the split counts come from
- [[per-head-background-reduction]] — why run4 counts frames twice
- [[mobilenetv4-sense-experiments]] — what was trained on these
- [[deviation-label-join-collision]] — a labelling bug in the Drive `sense` set
