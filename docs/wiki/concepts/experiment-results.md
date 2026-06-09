# Experiment Results

Sources: `dronet_results/summary.json`, `dronet_results/README.md`.

## Run Summary

The local run processed 12 readable clips and 480 total frames. Clip `01` / `006_sem_obstaculos.mp4` was unreadable and skipped.

Most readable clips are `2160x3840` portrait videos at about 60 FPS. One single-frame clip, `019_obstaculo_direita_parada`, is `1920x1080` at 30 FPS.

## Collision Probabilities

Collision probabilities are generally high:

- Several clips have `frac_collision_ge_0.5 = 1.0`.
- The lowest listed collision mean is clip `08`, at `0.6149`.
- The highest means are near 1.0, including clips `06`, `07`, and `09`.

This aligns with the domain-mismatch caveat in `dronet_results/README.md`: the model was trained on street-like car/bicycle views, not supermarket phone videos.

## Steering

Steering means are mostly modest but often negative:

- Strongest negative mean: clip `09`, `-0.2868`.
- Strongest positive mean: clip `03`, `0.0891`.
- Clip `019_obstaculo_direita_parada` has a single-frame steering value of `-0.7227`, so it should not be compared as a stable clip mean.

## Notable Data Issue

Clips `06` and `07` have identical aggregate values in `summary.json`:

- 26 frames.
- Steering mean `-0.1071`, min `-0.3329`, max `0.0281`.
- Collision mean `0.9899`, max `1.0`, fraction collision >= 0.5 of `1.0`.

This may be legitimate duplicate content or may indicate repeated/duplicated input clips. It is listed in [[open-questions]].

## Interpretation Boundary

These outputs are model predictions only. They do not establish that the original DroNet policy works in the supermarket environment.

## PilotGuru 10 FPS Folder-Name Check

Source: `dronet_results/pilotguru_10fps_match/summary.csv`. Detailed page: [[pilotguru-10fps-folder-check]].

On 2026-05-30, a 10 FPS pass sampled all 22 readable `.mp4` files under `C:\Users\Isaque\Downloads\Supermercado Telefrango (Sem GPS)\PilotGuru\exports`, including very short clips. It produced 463 sampled frames, saved annotated sample images under `dronet_results/pilotguru_10fps_match/sampled_frames/`, and wrote per-sample predictions to `samples.csv`.

The folder-name comparison used folder names as expected labels:

- `desvio_direita` -> expected RIGHT steering.
- `desvio_esquerda` -> expected LEFT steering.
- `centro_parada` / `parada` -> expected STOP, approximated from DroNet collision probability because DroNet has no explicit stop class.

With a steering threshold of `0.10` and collision threshold of `0.50`, 10 of 22 readable videos matched the folder-name expectation and 12 mismatched. All `obstaculo_esquerda_desvio_direita` clips mismatched; most were predicted LEFT or STOP instead of RIGHT. All `obstaculo_centro_parada` clips matched under the collision-based STOP criterion. Three `obstaculo_direita_desvio_esquerda` clips matched LEFT directly: `16`, `21`, and `22`.
