# PilotGuru 10 FPS Folder-Name Check

Source artifacts:

- `dronet_results/pilotguru_10fps_match/summary.csv`
- `dronet_results/pilotguru_10fps_match/summary.json`
- `dronet_results/pilotguru_10fps_match/samples.csv`
- `dronet_results/pilotguru_10fps_match/sampled_frames/`

## Purpose

This run sampled every readable `.mp4` under `C:\Users\Isaque\Downloads\Supermercado Telefrango (Sem GPS)\PilotGuru\exports` at 10 FPS and ran [[model-architecture|DroNet]] inference on each sampled frame. The goal was to compare the model's aggregate prediction for each video against the action encoded in the folder name.

The run processed 22 readable videos and 463 sampled frames. Short videos were included; each readable video received at least one sample when possible.

## Label Mapping

Folder names were converted to expected actions as follows:

- `desvio_direita` -> RIGHT.
- `desvio_esquerda` -> LEFT.
- `centro_parada` or `parada` -> STOP.

DroNet does not output a semantic STOP action. It outputs steering and collision probability. For this check, STOP was approximated by collision probability at least `0.50`.

## Prediction Rule

For each video, the sampled-frame steering and collision outputs were averaged.

- Steering greater than `0.10` -> RIGHT.
- Steering less than `-0.10` -> LEFT.
- Steering between `-0.10` and `0.10`, with mean collision probability at least `0.50` -> STOP.
- Steering between `-0.10` and `0.10`, with mean collision probability below `0.50` -> STRAIGHT.

For expected STOP clips, the match check used collision probability rather than the final steering-derived label, because the model has no explicit stop class.

## Aggregate Result

The folder-name check matched 10 of 22 readable videos and mismatched 12 of 22.

Category-level behavior:

- `obstaculo_esquerda_desvio_direita`: 0 of 7 matched. These clips expected RIGHT, but the model predicted LEFT for six clips and STOP for one.
- `obstaculo_centro_parada`: 7 of 7 matched under the collision-based STOP criterion.
- `obstaculo_direita_desvio_esquerda`: 3 of 8 matched. Clips `16`, `21`, and `22` matched LEFT; the others were mostly classified as STOP.

## Per-Video Summary

| Folder | Video | Expected | Predicted | Match | Samples | Duration s | Mean Steering | Mean Collision |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `01_obstaculo_esquerda_desvio_direita` | `01_obstaculo_esquerda_desvio_direita.mp4` | RIGHT | LEFT | mismatch | 10 | 0.9833 | -0.330397 | 0.679614 |
| `02_obstaculo_esquerda_desvio_direita` | `02_obstaculo_esquerda_desvio_direita.mp4` | RIGHT | LEFT | mismatch | 10 | 0.9833 | -0.144574 | 0.552738 |
| `03_obstaculo_esquerda_desvio_direita` | `03_obstaculo_esquerda_desvio_direita.mp4` | RIGHT | LEFT | mismatch | 10 | 0.9667 | -0.237455 | 0.574273 |
| `04_obstaculo_esquerda_desvio_direita` | `04_obstaculo_esquerda_desvio_direita.mp4` | RIGHT | LEFT | mismatch | 10 | 0.9667 | -0.384080 | 0.300011 |
| `05_obstaculo_esquerda_desvio_direita` | `05_obstaculo_esquerda_desvio_direita.mp4` | RIGHT | STOP | mismatch | 10 | 0.9667 | -0.079018 | 0.744786 |
| `07_obstaculo_esquerda_desvio_direita` | `07_obstaculo_esquerda_desvio_direita.mp4` | RIGHT | LEFT | mismatch | 10 | 0.9667 | -0.269420 | 0.712111 |
| `08_obstaculo_centro_parada` | `08_obstaculo_centro_parada.mp4` | STOP | STOP | match | 10 | 0.9667 | -0.083890 | 0.981657 |
| `09_obstaculo_centro_parada` | `09_obstaculo_centro_parada.mp4` | STOP | STOP | match | 10 | 0.9667 | -0.029427 | 0.982954 |
| `10_obstaculo_centro_parada` | `10_obstaculo_centro_parada.mp4` | STOP | STOP | match | 9 | 0.8667 | -0.001612 | 0.784643 |
| `11_obstaculo_centro_parada` | `11_obstaculo_centro_parada.mp4` | STOP | LEFT | match | 10 | 0.9667 | -0.100358 | 0.897151 |
| `12_obstaculo_centro_parada` | `12_obstaculo_centro_parada.mp4` | STOP | LEFT | match | 3 | 0.2833 | -0.157784 | 0.930619 |
| `13_obstaculo_centro_parada` | `13_obstaculo_centro_parada.mp4` | STOP | LEFT | match | 10 | 0.9500 | -0.120304 | 0.785963 |
| `14_obstaculo_centro_parada` | `14_obstaculo_centro_parada.mp4` | STOP | LEFT | match | 25 | 2.4833 | -0.111033 | 0.873979 |
| `15_obstaculo_direita_desvio_esquerda` | `15_obstaculo_direita_desvio_esquerda.mp4` | LEFT | STOP | mismatch | 15 | 1.5000 | -0.065619 | 0.886484 |
| `16_obstaculo_direita_desvio_esquerda` | `16_obstaculo_direita_desvio_esquerda.mp4` | LEFT | LEFT | match | 9 | 0.8167 | -0.119366 | 0.903927 |
| `17_obstaculo_direita_desvio_esquerda` | `17_obstaculo_direita_desvio_esquerda.mp4` | LEFT | STOP | mismatch | 8 | 0.8000 | -0.094038 | 0.893685 |
| `18_obstaculo_direita_desvio_esquerda` | `18_obstaculo_direita_desvio_esquerda.mp4` | LEFT | STOP | mismatch | 13 | 1.2833 | 0.047757 | 0.959182 |
| `19_obstaculo_direita_desvio_esquerda` | `19_obstaculo_direita_desvio_esquerda.mp4` | LEFT | STOP | mismatch | 11 | 1.0500 | -0.032944 | 0.912679 |
| `20_obstaculo_direita_desvio_esquerda` | `20_obstaculo_direita_desvio_esquerda.mp4` | LEFT | STOP | mismatch | 14 | 1.3500 | -0.068763 | 0.941906 |
| `21_obstaculo_direita_desvio_esquerda` | `21_obstaculo_direita_desvio_esquerda.mp4` | LEFT | LEFT | match | 125 | 12.4333 | -0.480334 | 0.227085 |
| `22_obstaculo_direita_desvio_esquerda` | `22_obstaculo_direita_desvio_esquerda.mp4` | LEFT | LEFT | match | 118 | 11.7333 | -0.155091 | 0.578518 |
| `23_obstaculo_esquerda_desvio_direita` | `23_obstaculo_esquerda_desvio_direita.mp4` | RIGHT | STOP | mismatch | 13 | 1.2833 | -0.058325 | 0.846927 |

## Extremes

- Longest sampled clip: `21_obstaculo_direita_desvio_esquerda`, 125 samples over 12.4333 s.
- Shortest sampled clip: `12_obstaculo_centro_parada`, 3 samples over 0.2833 s.
- Most negative mean steering: clip `21`, `-0.480334`.
- Most positive mean steering: clip `18`, `0.047757`; no clip had a positive mean steering above the RIGHT threshold.
- Highest mean collision: clip `09`, `0.982954`.
- Lowest mean collision: clip `21`, `0.227085`.

## Interpretation Limits

This is a model-output comparison against folder-name conventions, not a formal accuracy measurement. The folder names encode expected behavior at the clip level, but the wiki has not verified independent frame-level ground truth labels. The model was trained on outdoor navigation imagery, and these supermarket phone videos are out of distribution. See [[open-questions]] for validation gaps.
