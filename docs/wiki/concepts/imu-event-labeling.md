---
type: concept
tags: [imu, labeling, events, sensor, methodology]
code_refs: [src/sensecv/app.py, scripts/imu_event_report.py]
updated: 2026-06-12
---

# IMU event labeling (criterios PDF)

Implementation of the advisor's methodology document
`data/uploaded_datasets/criterios_imu_video_orientando.pages.pdf`: validate the
IMU capture, locate physical events (desvio, reducao, parada) with T1/T2 from
the IMU, and label the **decision window before the movement** rather than the
maneuver itself. Golden rule: *the IMU says when the body reacted; the video
shows why; the training label must point to the frames that precede or start
the reaction.*

## Capture validation (PDF section 1)

`app.py › imu_quality_report(idx)` runs the checklist and returns a verdict:

| check | pass condition |
|---|---|
| `taxa_amostragem_100hz` | accel median rate ≥ 100 Hz |
| `sem_lacunas` | no inter-sample gap > 3× median interval |
| `jitter_baixo` | dt std / median dt < 0.5 |
| `sem_saturacao` | < 1% of samples pinned near the absolute max |
| `marcha_periodica` | gait autocorrelation peak ≥ 0.3 with cadence in 0.8–3.3 Hz |
| `tem_caminhada` | > 20% of frames above the walking energy threshold |
| `estados_distinguiveis` | p90/p10 energy ratio > 2 or a visible stopped stretch |

Verdict: `aceitar` (all pass), `baixa_confianca` (minor fails), `rejeitar`
(saturation, no walking, or no gait periodicity — the critical set).

## Event detection (PDF sections 2–3)

`app.py › detect_imu_events(idx, delta)` builds frame-aligned signals in
`_imu_event_series()` and emits events:

- **desvio** — persistent excursion of a lateral signal, with single isolated
  peaks rejected (persistence ≥ 0.4 s at 60% of the threshold, peak above the
  full threshold). Out-and-back excursions of opposite sign within 1.5 s merge
  into one event: T1 = first excursion onset, T2 = return to trajectory, and
  the **strongest** excursion carries the direction. Signal choice per clip:
  gravity-projected gyro **yaw rate** (`> 25 °/s`) when the capture actually
  rotates, else **lateral velocity** — gravity-orthogonal accel integrated
  with rolling-mean drift removal — because gimbal-stabilized captures barely
  yaw. Positive = esquerda (right-hand rule around the gravity-aligned up
  vector, left = up × forward with forward = camera −z projected horizontal).
- **reducao** — walking energy persistently below 60% of its rolling 3 s
  baseline without reaching standstill.
- **parada** — gait energy below 0.25 m/s² for ≥ 1 s (`SENSECV_IMU_STOP_MIN_SEC`).

## Label windows (PDF sections 2.2 / 4)

Every event carries the three windows, with Δ defaulting to 1.0 s
(`SENSECV_IMU_EVENT_DELTA_SEC`, UI options 0.5 / 1.0 / 1.5):

| window | range | use |
|---|---|---|
| `acao` | T1 → T2 | the maneuver itself |
| `decisao` | T1−Δ → T1 | scene that demanded the action — preferred for CNN training |
| `expandido` | T1−Δ → T2+0.3s | when sync or reaction onset is uncertain |

## Confidence (PDF sections 1.2 / 5)

`alta` requires: capture verdict `aceitar`, signal margin ≥ 1.5× threshold,
and no overlapping event of a different type (a reducao flowing into a parada
is treated as a natural pair, not ambiguity). Capture `rejeitar` ⇒ all events
`descartar`. Everything else ⇒ `baixa` with explicit `confidence_reasons`.

## Validation against the IFCE manifest

On the 32 clips of [[sensecv-02062026-ifce-clip-manifest]] (one known
deviation each): **26/32 correct directions, 3 misses, 3 wrong** — and 2 of
the 3 wrong are flagged `baixa`. The yaw path is untested on this dataset
because the gimbal suppresses rotation (max ≈ 11 °/s); lateral velocity was
chosen automatically for all 32. No paradas are detected because recording
stops less than 1 s after the person stops.

## Surfaces

- `GET /api/imu-events/<idx>?delta=` — see [[api-routes]].
- Viewer "Eventos IMU" panel — Δ selector + Detectar button; each event shows
  type, direction, T1/T2, confidence, and three window buttons that apply the
  range as the crop window. See [[viewer-frontend]].
- `scripts/imu_event_report.py` — batch CSVs (`quality.csv`, `events.csv`)
  under `data/derived/imu_event_report/`.

Section 9 of the PDF (similarity-based frame cuts instead of fixed FPS) is the
metric frame selection documented in [[clear-dataset]] and `/api/ssim`.
