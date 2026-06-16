---
type: entity
tags: [dataset, metadata, SenseCV, IFCE, manifest]
code_refs: []
updated: 2026-06-10
---

# SenseCV 02-06-2026 IFCE clip manifest

Human-authored spreadsheet metadata for the uploaded
`SenseCV-02-06-2026-IFCE-Gimbal` dataset. This is the best source for answering
"what kind of clip is ID NN?" before looking at the video itself.

Source workbook:
`data/uploaded_datasets/SenseCV-02-06-2026-IFCE-Gimbal/SenseCV-02-06-2026-IFCE-Gimbal/Coleta  IFCE - 02 de junho de 2026 - ANDRE.xlsx`

The path above is ASCII-normalized for wiki readability; the actual local
filename uses the accented final `E` in the author name. Only one `.xlsx` file
was found under `data/uploaded_datasets/`.

## Workbook shape

- Sheet: Portuguese `Pagina1`, with the accent stored in the workbook metadata.
- Clip rows: 32 non-empty IDs, matching dataset folders `01` through `32`.
- Columns:
  - `ID`
  - `LOCAL`
  - `DESCRICAO`
  - `POSICAO CELULAR`
  - `LOCAL OBSTACULO`
  - `ALTURA OBSTACULO`

All clips are recorded in the corridor in front of the Huawei lab. The repeated
scenario text says the operator starts at 3.5 m, places an obstacle to the left
or right, deviates at 1.5 m from the obstacle, and stops 1.5 m after the
obstacle.

## Dataset balance

| axis | values |
|---|---|
| phone pose | 28 `celular em pe`, 4 `celular deitado` |
| obstacle side | 16 right, 16 left |
| deviation side | 16 left, 16 right |
| obstacle level | 20 floor/chao, 12 high/alto |
| obstacle height | 20 blank floor rows, 8 at 1.30 m, 4 at 1.10 m |
| objects | 8 blue trash bin, 8 earphone box, 4 microphone box, 8 gimbal box, 4 extinguisher |

Obstacle side and deviation side are paired opposites:
right-side obstacles use a left deviation, and left-side obstacles use a right
deviation.

## Clip ID map

| IDs / folders | object | phone pose | obstacle level | height | obstacle side | deviation |
|---|---|---|---|---|---|---|
| `01`, `02` | blue trash bin | lying/deitado | floor | - | right | left |
| `03`, `04` | blue trash bin | lying/deitado | floor | - | left | right |
| `05`, `06` | blue trash bin | upright/em pe | floor | - | right | left |
| `07`, `08` | blue trash bin | upright/em pe | floor | - | left | right |
| `09`, `10` | earphone box | upright/em pe | floor | - | right | left |
| `11`, `12` | earphone box | upright/em pe | floor | - | left | right |
| `13`, `14` | microphone box | upright/em pe | floor | - | right | left |
| `15`, `16` | microphone box | upright/em pe | floor | - | left | right |
| `17`, `18` | gimbal box | upright/em pe | floor | - | right | left |
| `19`, `20` | gimbal box | upright/em pe | floor | - | left | right |
| `21`, `22` | earphone box | upright/em pe | high | 1.30 m from floor | right | left |
| `23`, `24` | earphone box | upright/em pe | high | 1.30 m from floor | left | right |
| `25`, `26` | gimbal box | upright/em pe | high | 1.30 m from floor | right | left |
| `27`, `28` | gimbal box | upright/em pe | high | 1.30 m from floor | left | right |
| `29`, `30` | extinguisher | upright/em pe | high | 1.10 m from floor | right | left |
| `31`, `32` | extinguisher | upright/em pe | high | 1.10 m from floor | left | right |

## Interpretation notes

- Folder ID and spreadsheet `ID` correspond directly, zero-padded on disk.
- The spreadsheet describes the physical collection setup. It is not the same
  as the classified export labels in [[exports-output]] or [[history-json]].
- The raw uploaded dataset is the source used by the deployed example dataset
  described in [[deployment-operations]] and by the DroNet sampling pass in
  [[dronet-sensecv-02062026-3fps]].
