# Wiki Log

Append-only. One entry per operation. Greppable prefix:
`grep "^## \[" log.md | tail -5`.

---

## [2026-08-17] experiment | DINOv2 fine-tuning ladder and campaign ingest
- Trained DINOv2 ViT-S/14 two-head on the exact fps5/uniao data path: `ultimos1`
  (1.78M trainable, 8.1%), `ultimos2` (3.55M, 16.1%) and `completo` (22.06M).
- `ultimos1` is best: T1 joint 0.8962, T4 0.9194 — the best T4 of any model so
  far, beating MobileNetV4 full_rate/uniao (0.8607) while training on 8,029
  frames instead of 34,394. Unfreezing a second block does not help.
- `completo` scored 0.4426 on T1 but did not diverge: it was still converging
  when patience ran out. `lr 1e-4` (run4's) is ~10x too high for a full ViT-S
  fine-tune; the rerun at 1e-5 is an open thread.
- Moved every script, result, checkpoint and DINOv2 selection out of session
  scratch into `scripts/` and `Desktop\sense_run4_experiments\`, and repointed
  the copied scripts at the durable paths.
- Pages touched: [[dinov2-sense-finetuning]] (new), [[sense-experiments-state]]
  (new), [[mobilenetv4-sense-experiments]], [[index]].

## [2026-08-16] experiment | MobileNetV4 sense datasets, variants and evaluation matrix
- Built four datasets in the Drive `sense` tree/label style: the bit-exact run4
  reproduction (33,118 / 6,172 / 3,970), distinct frames (24,347), distinct +
  flip (48,694), and left/right clips with the window extended 0.5 s past the
  cut (10,047 / 1,930 / 2,706, 327 of 328 clips resolved).
- Trained 8 MobileNetV4 two-head models on Kaggle across four variants (full
  rate, 5 FPS, DINOv2 0.98, held-out-day validation) x {baseline, uniao}, all on
  run4's configuration but with both heads trained on every frame.
- Scored all 8 on 6 test sets; ranking is stable, absolute scores move up to 15
  points with the test set. full_rate/uniao leads everywhere.
- Key finding: clip-grouped validation gives val loss 0.0003 versus 0.275 for a
  genuinely held-out day, and the union's advantage does not survive the honest
  split. Documented as leakage evidence for the paper.
- Pages touched: [[sense-style-datasets]] (new), [[allframes-split-provenance]]
  (new), [[per-head-background-reduction]] (new),
  [[mobilenetv4-sense-experiments]] (new), [[index]].

## [2026-08-16] query | deviation label join collision in the ConvLSTM datagen
- Traced the paper's frame table to `*_two_head_all_frames.py`: pool 52,719
  frames / 608 clips (dataset8 22,781 + dataset9 814 + auto-trim 29,124), fixed
  day-22/24 test of 3,970, clip-grouped validation, then per-head background
  reduction; the reported train/val columns are `*_reduced_per_head`.
- While cross-checking against `sense/` on Drive, found that
  `convlstm_mobilenetv1_all_frames.py › read_lateral_labels()` joins on
  `(source_clip, frame_index)`, which is not unique once `_head`/`_tail`
  suffixes are stripped. Last-wins relabels 2,267 `left` and 2,270 `right`
  frames as `none`; the test split is unaffected.
- Reproduced run4's MobileNetV1 dataset bit-exactly (33,118 / 6,172 / 3,970)
  and materialized it outside the repo, in the `sense` file tree and label
  style.
- Pages touched: [[deviation-label-join-collision]] (new), [[index]].

## [2026-08-12] feature | magnetometer chart in the viewer player
- `get_clip_data()` now loads `magnetometer.json` (when present) and adds a
  frame-interpolated `magnetometer` field to `/api/data`, alongside `accel`
  and `rotation`.
- Viewer gains a fourth chart (`#block-magnetometer`, blue, X/Y/Z/`|M|`
  per-series toggles) between Giroscópio and Velocidade, built the same way
  as the acceleration/gyroscope charts and included in the shared zoom sync.
  It self-hides for clips without a magnetometer capture.
- Pages touched: [[viewer-frontend]], [[api-routes]].

## [2026-07-27] feature | save the DINO endpoint training subset
- Added `scripts/save_dinov2_endpoints_train_dataset.py` to reproduce the
  endpoint trainer's current selection/split and copy only its train images.
- The portable ImageFolder-style output includes `labels.csv` provenance and
  `split_summary.json` with class/frame/clip counts for all three splits.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-26] feature | MobileNetV1 endpoint initialization comparison
- Added `scripts/train_mobilenetv1_endpoints_kaggle.py` for a same-split
  ImageNet-versus-random-initialization MobileNetV1 comparison.
- It shares the DINO endpoint trainer's first/last sequence selection,
  clip-grouped train/validation split, fixed day-22 test set, undated-clear
  fallback, and per-class split counts.
- It saves both best-validation checkpoints and one comparison metrics JSON.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-26] improve | endpoint split class-count summary
- The endpoint DINOv2 trainer now prints the number of frames in every class
  for train, validation, and test immediately after creating the split.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-26] correction | undated clear clips belong to day 22
- The endpoint DINOv2 trainer now assigns every undated `sem_obstaculos` /
  `source=clear` sequence to a `SenseCV-22-06-2026-clear_*` clip ID before
  splitting, preserving the dataset rule even if `labels.csv` omits the date.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-26] feature | minimal endpoint-only DINOv2 Kaggle trainer
- Added `scripts/train_dinov2_endpoints_kaggle.py`, a single-cell, `CONFIG`-only
  pipeline over one existing all-frames dataset.
- It keeps only the first and last frame of each decoded sequence, preserves
  the clip-grouped validation split and fixed day-22 test set, caches frozen
  DINOv2 embeddings, and trains only one dense classification layer.
- Output is limited to `model.pt`, `metrics.json`, and two console messages;
  the existing full DINOv2 trainer remains unchanged.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-22] fix | bundle the required DINO dataset builder
- `train_dinov2_classifier.py` now embeds an integrity-checked copy of the
  labelled-MP4/day-22-clear builder and uses it whenever no compatible sibling
  script is available. A single pasted/uploaded trainer now works on Kaggle.
- Compatible local companions still override the bundled copy; incompatible
  older image-folder builders are ignored without changing dataset provenance.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-22] feature | standalone DINOv2 test evaluator
- Added `scripts/evaluate_dinov2_test.py` to restore promoted DINOv2 `model.pt`
  checkpoints and evaluate the exact fixed day-22 frames reconstructed from
  each dataset's `labels.csv`.
- Outputs per-dataset metrics/confusion/per-class reports and per-frame labels,
  confidence, and probabilities, plus a combined summary. Evaluation is
  unaugmented and unweighted.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-22] correction | require the labelled-MP4 DINO builder
- The DINOv2 classifier now accepts only the builder that decodes labelled
  `Todos_cortes_auto_trim/deviation` MP4 cuts, uses `cut_info.json` with
  `sources.csv` fallback, and adds DINO-selected day-22 clear images.
- The older published image-folder companion is rejected rather than silently
  creating a dataset with different provenance. Upload both current scripts to
  Kaggle until the matching companion contract is published.
- Build defaults are threshold `0.96`, cap 25 frames per clip, and seed `43`;
  the MobileNet companion defaults to `clip_balanced` only.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-22] fix | Kaggle DINOv2 companion bootstrap
- `train_dinov2_classifier.py` now searches Kaggle attachments and, for a
  pasted standalone cell, downloads `train_mobilenet_dino.py` from the public
  SenseCV repository into `/kaggle/working` instead of failing during import.
- Added compatibility defaults and local classification metrics so both the
  published image-dataset companion and newer video-dataset revision load.
  Internet-enabled auto-download and explicit two-file upload are supported.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-21] feature | DINOv2 single-head classification trainer
- Added `scripts/train_dinov2_classifier.py`, which reuses the existing DINO
  dataset builder and trains pretrained DINOv2 ViT-S/14 or ViT-B/14 classifiers.
- Preserved clip-grouped splits, fixed day-22 testing, canonical class order,
  label-safe augmentation, train-only balancing, validation-only promotion,
  and per-trial reports. The default sweep compares a linear probe with last-one
  and last-two transformer-block fine-tuning.
- Promoted checkpoints are `model.pt` files with the backbone, class,
  preprocessing, and state-dict metadata; they are not yet viewer-loadable.
- Pages touched: [[kaggle-image-datasets]], [[index]].

## [2026-07-21] feature | day-22 clear frames in DINO test datasets
- The DINO builder now considers every `*sem_obstaculos*.jpg` from dataset8's
  `deviation_lrn/none` folder, groups all 1,740 source images by 58 day-22
  `source_clip` values, and runs the normal DINOv2 selector independently per
  clip. Only selected frames enter both outputs as `free_path` / `none`; frame
  indices, timestamps, and clip provenance come from the source `labels.csv`.
- Dataset freshness compares source count plus DINO threshold/cap, so older or
  differently selected outputs rebuild automatically. `--day22-clear-dir ""`
  disables the addition and `--no-build` still skips all building.
- The fixed test day returns to 22 now that it has clear `free_path`/`none`
  examples alongside obstacle/left/right cuts.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-21] feature | selectable pretrained MobileNetV1–V4 DINO trainer
- `scripts/train_mobilenet_dino.py` now accepts `--mobilenet-version v1|v2|v3|v4`
  and fixes every option to pretrained ImageNet weights at baseline width
  (`alpha=1.0`) and 224×224; incompatible values and weight-load failures stop
  explicitly instead of falling back to training from scratch.
- V1/V2/V3 use Keras Applications and promote `model.keras`; V4 uses timm's
  `mobilenetv4_conv_small.e2400_r224_in1k` PyTorch preset and promotes
  `model.pt`, while retaining the clip split, weighting, fine-tuning trials,
  validation-only model selection, and report schema.
- Documented the current trimmed-video DINO build input and the V4 viewer-format
  limitation.
- Pages touched: [[kaggle-image-datasets]], [[index]].

## [2026-07-14] improve | DINO MobileNet fine-tuning strategy sweep
- `scripts/train_mobilenet_dino.py` now compares three reproducible recipes on
  one clip-grouped split: the original full-network baseline, conservative
  last-30-layer tuning with frozen BatchNorm, and last-45-layer AdamW tuning
  with class+clip-balanced weights. Strong trials use label-safe crop/color
  augmentation and still never horizontally flip left/right labels.
- Fixed best-model handling: the global best validation-loss checkpoint is
  reloaded after both phases instead of being overwritten by the final epoch.
  Validation/test weights are now 1 regardless of train weighting.
- Each trial writes its own model/report/history with validation and test
  macro-F1/balanced accuracy. The deployable model is promoted strictly by
  validation macro-F1; day-22 test results are not used for selection.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-03] feature | margin-bounded "trim" cut mode (2s from each end)
- New way of choosing cuts, requested by the operator (refined thrice
  in-session: whole-middle `[2s, dur-2s]` → clamp-shifted → search-bounded →
  final +8 s exemption/parada revert): **desvio** cuts (~1 s, same PDF window)
  are only *looked for* inside `[2s, duration-2s]` — edge events are
  discarded, never shifted. **Clips under 8 s** (`TRIM_CUT_MIN_DURATION_SEC`,
  env-overridable) skip the margins and return exactly the plain deviation
  cut; **parada** (1 s stop-onset window, full-clip search) and **livre**
  (whole clip) are untouched in every case.
- `app.py`: `suggest_deviation_cut` gains `cut_margin=None` (filters desvio
  events by their cut window, bounds the manifest signal recovery via new
  `t1_min`/`t1_max` params on `_desvio_window_from_signal`, self-disables
  under 8 s); `suggest_trim_cut(idx, margin=None)` is a thin wrapper adding
  `mode:'trim'`, `trimmed:true`, `margin`. Margin env
  `SENSECV_TRIM_CUT_MARGIN_SEC` (default 2.0 s/side).
- Exposed at `/api/suggest/<idx>?mode=trim` (same SSIM wrapper) and as a third
  viewer button "⬦ Miolo 2s"; the cut-type chip shows the resolved class with
  a `· miolo 2s` suffix. After the operator hit a pre-margin cut on clip
  viewer-284/idx-283 (on-load auto-suggest still used `lateral`),
  `runAutoSuggest()` now tries **trim → walking**, so the margins are the
  default; the "⬦ Desvio lateral" button keeps the unbounded cut.
- Verified live (150 of 398 clips): 30 sub-8 s clips identical to the plain
  cut; parada keeps 17.967–18.967; long-clip desvios 0 margin violations
  (2 edge desvios re-detected in-margin).
- Pages touched: [[crop-suggestion]], [[api-routes]], [[viewer-frontend]].

## [2026-07-01] add | test-set prediction dump script for the trained models
- `scripts/predict_test_frames.py`: standalone Kaggle companion run after any
  of the trainers. Rebuilds the day-`--test-day` test set from labels.csv,
  loads each `<models-root>/<dataset>/model.keras`, predicts all test frames
  and writes `--out/<dataset>/`: images copied into one folder per PREDICTED
  label (`ok__<name>` when correct, `true-<label>__<name>` when wrong),
  `predictions.csv` with per-class probabilities, and `summary.json`
  (accuracy, per-true-label recall + predicted_as). Class order from the
  model dir's report.json (canonical fallback).
- Defaults target the _dino datasets + models_dino; `--datasets ""` sweeps
  every dataset that has a model. Compile-checked only (TF is Kaggle-side).
- Pages touched: [[kaggle-image-datasets]], [[index]].

## [2026-07-01] change | train_mobilenet_dino.py now builds its own datasets
- The DINO trainer is now a one-file build+train pipeline over the uploaded
  **all-frames 224 image datasets** (`--input` default
  `/kaggle/input/datasets/isaqueefraim/dataset8/kaggle_allframes_datasets_224`):
  no video decoding — it groups `obstacle_freepath/labels.csv` by physical
  frame sequence (filename base before `__fNNNNN.jpg`), runs `Dinov2Selector`
  per sequence, and copies the kept frames into `obstacle_freepath_dino` +
  `deviation_lrn_dino` (both schemes matched by filename, all columns incl.
  `source`/`segment` preserved) before training at img_size 224.
- Build auto-skips when both `labels.csv` exist; `--rebuild` forces it,
  `--no-build` restores the old train-only behaviour. New flags: `--input`,
  `--dino-threshold`, `--dino-max-per-clip`, `--limit`.
- Verified locally on 5 clear sequences of the local 224 build: 17/94 frames
  kept, both schemes + labels.csv/dataset_info.json written; the trainer half
  is unchanged (TF not installed locally, exercised on Kaggle).
- Pages touched: [[kaggle-image-datasets]], [[index]].

## [2026-07-01] add | standalone trainer for the all-frames 224 dataset
- `scripts/train_mobilenet_allframes.py`: paste-friendly standalone copy of the
  MobileNetV1 trainer (like `train_mobilenet_dino.py`) defaulted to the
  uploaded all-frames dataset (`obstacle_freepath,deviation_lrn` straight from
  /kaggle/input), img_size 224, out `models_allframes/`. Report/print gain a
  `test_frames_by_source` breakdown (cuts vs clear on the test day).
- Split dry-run on the local 224 build: obstacle scheme trains 17 952 / val
  3 739 / test 3 156 frames over 308/55/91 clips; test = 1 416 cuts + 1 740
  clear frames. Deviation scheme test: left 850 / right 566 / none 1 740.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-01] change | all-frames dataset rebuilt at 224x224
- `make_kaggle_allframes_dataset.py` gained `--img-size` (default 224, 0 =
  native 720x720); frames are resized with INTER_AREA on write.
- Rebuild at `data/derived/kaggle_allframes_datasets_224/`: same 893 clips /
  24 847 images per scheme, now 224x224x3, ~0.9 GB (was ~6 GB native). The
  old native-resolution `kaggle_allframes_datasets/` was deleted.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-01] fix | day-22 test split was missing most of its clips
- Reported: the trainer's day-22 test set only saw a scattered subset of the
  day-22 clips. Root causes were in the DATASET, not the trainer: (1) 136 of
  396 `datasetcortado-andre` cut folders (incl. 23 of the 33 day-22 clips)
  have no per-folder `cut_info.json` — their labels live only in the root
  `sources.csv` — so the builder dropped them; (2) the `test-clear` clips are
  the day-22 free-path recordings but their `NN_sem_obstaculos` names carry no
  date, so the trainer's `DD-MM-YYYY` parser routed them to train/val.
- `make_kaggle_allframes_dataset.py`: cut folders now fall back to the root
  `sources.csv` ledger for labels; new `--clear-date` (default `22-06-2026`)
  prefixes `SenseCV-<date>-clear_` onto clear `source_clip`s.
- Rebuild: 893 clips (396+439+58, 0 skipped), 24 847 images per scheme; the
  day-22 test split is now 91 clips / 3 156 frames with both classes
  (obs 1 416 / free 1 740; left 850 / right 566 / none 1 740).
- Remaining gap: `dataset_endpoints` predates the day-22 uploads (0 day-22
  folders) — rebuild via `make_dataset_endpoints.py` if day-22 endpoint frames
  are wanted.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-01] feature | combined all-frames Kaggle dataset from three sources
- `scripts/make_kaggle_allframes_dataset.py`: self-contained builder merging
  every frame from (1) the labelled `datasetcortado-andre` deviation cuts,
  (2) the `dataset_endpoints` head/tail 0.5 s cuts, and (3) the `test-clear`
  `sem_obstaculos` clips into the two single-head schemes
  (`obstacle_freepath`, `deviation_lrn`). Endpoints/clear frames are labelled
  `free_path`/`none`; `labels.csv` gains `source` + `segment` columns;
  `source_clip` strips `_head`/`_tail` so clip-grouped splits don't leak.
- Two build fixes found on the first run: the endpoints root's
  `review_all_frames.mp4` montage was being ingested as 11 710 bogus free_path
  frames (root-level `review*.mp4` now excluded), and the one endpoints folder
  missing `cut_info.json` was silently dropped (segment now inferred from the
  folder suffix).
- Build @ `data/derived/kaggle_allframes_datasets/`: 757 clips (260+439+58),
  136 unlabelled cut folders ledgered in `skipped.json`; 18 905 images per
  scheme (obs: 11 687 free_path / 7 218 obstacle; dev: 12 390/3 609/2 906),
  ~4.9 GB.
- Pages touched: [[kaggle-image-datasets]], [[index]].

## [2026-06-28] add | dedicated MobileNet-on-DINOv2-datasets trainer + preprocessing fix
- `scripts/train_mobilenet_dino.py`: standalone (self-contained, paste-friendly)
  copy of the trainer defaulting `--datasets` to
  `obstacle_freepath_dino,deviation_lrn_dino` and `--out-root` to `models_dino/`.
- `train_mobilenet_datasets.py`: **fixed input contract** so the saved `.keras`
  works in the live viewer — moved normalisation INTO the model
  (`Rescaling(1/127.5,-1)` first layer) and feed raw 0-255 in the data pipeline,
  matching `app.py _sensemodel_preprocess` (which feeds raw 0-255). Models trained
  before this fix must be retrained.
- Pages touched: [[kaggle-image-datasets]].

## [2026-06-28] feature | single-head .keras models in the live viewer + activation maps
- The live inference (`sensemodel_frame_classification`) and Grad-CAM
  (`_sensemodel_activation`) paths now **auto-detect single-head models**: heads
  are matched by output width (2=obstacle, 3=deviation) and a head the model does
  not expose comes back `None`/`{available:false}` instead of being faked from the
  wrong output. `_split_head_arrays` returns `None` for an absent head.
- `_sensemodel_info()` / `/api/sensemodel-info` now report the detected `heads`
  list once a model is loaded; `sensemodel_frame_classification` adds `heads` and
  nulls the absent head's label/prob/probs.
- Frontend `renderSensemodel` guards the null head (`—`); the activation panel
  already degraded per-map, so it shows "cabeça ausente" for the missing head.
- `train_mobilenet_datasets.py` now emits a **canonical class order**
  (`free_path,obstacle` and `left,right,none`) so a single-head model's argmax
  lines up with `SENSECV_OBSTACLE_LABELS` / `SENSECV_DEVIATION_LABELS`.
- Pages touched: [[activation-maps]], [[kaggle-image-datasets]]. Relates to
  [[dronet-live-classification]], [[api-routes]].

## [2026-06-28] feature | self-contained Kaggle single-head datasets + MobileNetV1 trainer
- `scripts/make_kaggle_image_datasets.py`: builds four ImageFolder datasets
  (`obstacle_freepath`, `deviation_lrn`, + `_dino` variants) straight from the
  `datasetcortado/deviation` cut folders — only `cut_info.json` + the trimmed
  `<clip>.mp4`, no `sensecv.app`/IMU. DINOv2 ViT-S/14 cosine dedup (thr 0.98) for
  the `_dino` sets. First real build: 259 clips → 10613 + 1670 images each scheme.
- `scripts/train_mobilenet_datasets.py`: MobileNetV1 α=0.5 @128 ImageNet trainer,
  one model per dataset. Clip-grouped split (no frame leakage), **test = day 22**
  (`--test-day`), stratified train/val over the rest, inverse-freq class weights,
  horizontal-flip auto-disabled for the left/right deviation set. Two-phase
  warmup + fine-tune, writes confusion matrix / per-class F1 / history.
- Both notebook-friendly (top-of-file `CONFIG`, `parse_known_args`).
- Pages touched: [[kaggle-image-datasets]] (new), [[index]]. Relates to
  [[kaggle-two-head-dataset]], [[clear-dataset]], [[classification-taxonomy]],
  [[crop-suggestion]]. Page count: 40.

## [2026-06-25] refine | tighten stop and free-walk criteria
- **Stop cut** now requires a *confirmed* halt: the standstill must be preceded
  by walking, **not walked out of**, and last `≥ SENSECV_IMU_STOP_CONFIRM_SEC`
  (0.8 s). Dropped the old "≥ 0.3 s *or* reaches clip end" leniency, which let
  brief mid-walk pauses through. `app.py › detect_stop_onset`.
- **Free-walk cut** now requires a *long continuous walking run*: gait runs split
  by gaps `< SENSECV_IMU_WALK_MERGE_GAP_SEC` (0.3 s) are merged and the longest
  must last `≥ SENSECV_IMU_FREE_WALK_MIN_SEC` (3.0 s). `detect_free_walk_span`.
- Finding: on the current 389-clip dataset the confirmed-stop rule yields **0
  paradas** — captures end mid-stride, so a settled standstill is never recorded;
  the old paradas were sub-second pauses. New distribution: **162 desvio, 0
  parada, 97 livre, 130 none**.
- pages touched: [[crop-suggestion]], [[imu-event-labeling]]

## [2026-06-25] feature | free-walk fallback when no desvio and no parada
- A clip with **neither a desvio nor a parada** previously made
  `suggest_deviation_cut` return `found:false`. It is now treated as a **free
  walk** (caminhada livre) — unobstructed forward walking, a legitimate third
  dataset class alongside desvio and parada.
- `app.py`: new `detect_free_walk_span(idx)` returns the **whole usable walking
  span** `[first_walk, last_walk]` over the gait mask (`std_mag > IMU_WALK_STD`).
  `suggest_deviation_cut`'s final branch now cuts that span and returns
  `found:true` with `event_type:'livre'`, `side:'NONE'`, `direction:None`. Only a
  clip that never really walks still yields `found:false`.
- `export_set(mode='deviation')` now exports these cuts; `export_deviation_set`
  still filters to `side ∈ {LEFT, RIGHT}`, so `livre`/`parada` stay out of the
  desvio validation video. Auto-suggest now settles on `lateral` for any clip
  with walking; the viewer toast tags `livre` cuts "caminhada livre".
- pages touched: [[crop-suggestion]], [[imu-event-labeling]]

## [2026-06-23] feature | deviation cut falls back to stop-onset when no desvio
- Clips with **no lateral deviation** previously made `suggest_deviation_cut`
  return `found:false` (auto-suggest then fell through to walking). Now they cut
  around **the moment the person starts to stop**.
- `app.py`: new `detect_stop_onset(idx)` — onset of the last standstill stretch
  (`std_mag < IMU_STOP_STD`) that is preceded by walking and either lasts
  `≥ SENSECV_IMU_STOP_ONSET_MIN_SEC` (new env, default 0.3 s) **or runs to the
  end of the clip**. Looser than the strict `parada` event (`IMU_STOP_MIN_SEC`
  1.0 s), which rarely fires because recordings end < 1 s after the halt. The
  crossing into standstill is **T1** = the exact stop start.
- `suggest_deviation_cut` now falls back to this when no `desvio` qualifies,
  cutting the same PDF label window around T1 (`decisao` → `[T1-Δ, T1]`) and
  returning `event_type:'parada'`, `stop_time` (=T1), `side:'NONE'`. The desvio
  path gained `event_type:'desvio'`. `found:false` only when neither exists.
- `index.html`: the manual "Desvio lateral" toast now shows `parada @ <stop>`
  with the exact stop time for fallback cuts; `runAutoSuggest`/export/inspection
  pick up the cut transparently via the unchanged return shape.
- Verified: full scan of 389 clips → 166 stop-onset cuts, 61 still no-cut;
  desvio clip 0 unchanged (LEFT 1.471→2.471); parada clip 41 stop_time 6.233 →
  cut 5.233→6.233; `/api/suggest?mode=lateral` 200 with ssim; `/` renders 200.
- Pages touched: [[crop-suggestion]], [[imu-event-labeling]], [[log]].

## [2026-06-19] feature | new "Ativação" tab — Grad-CAM maps for the three heads
- Added a seventh viewer tab showing **Grad-CAM activation heatmaps** for the
  frame currently in the player: DroNet's **collision (classification) head**,
  and the SenseCV `.keras` two-head model's **obstacle (head 1)** and **deviation
  (head 2)** heads, side by side.
- `app.py`: new `/api/activation/<idx>?time=&exact=` route + `activation_maps_frame()`
  (decodes the frame once, shares the dronet/sensemodel 3-fps/exact frame logic
  and a 24-entry cache). Grad-CAM helpers: `_dronet_activation()` (PyTorch hooks
  on `conv9`, backprop the collision output), `_sensemodel_activation()` (TF
  `GradientTape` over the last 4D-output conv layer, one CAM per head),
  `_keras_last_conv_layer()`, and `_cam_to_data_uri()` (JET colormap blended over
  the exact model input, returned as a PNG data URI). Each sub-map degrades
  independently if torch/TF is missing.
- `index.html`: `ativacao` tab/view (video-centric, player visible for scrubbing),
  `requestActivation()`/`renderActivation()`/`resetActivationPanel()` driven from
  inside `requestDronet()` but **gated to only fetch when the tab is active**
  (the maps are expensive); `'ativacao'` added to `VIEWS`; `setView('ativacao')`
  triggers an exact-frame request.
- Verified: `/api/activation/0?time=1.0&exact=1` → 200, all three maps
  `available:true` with valid PNG data URIs (dronet CLEAR 0.0002, obstacle
  OBSTACLE 0.985, deviation LEFT 0.97 on frame 30).
- Pages touched: [[viewer-frontend]], [[api-routes]], [[activation-maps]] (new),
  [[index]].

## [2026-06-19] feature | split frame selection into its own large "Frames" tab
- `index.html`: frame-selection UI (métrica select, threshold slider/number,
  summary, `#ssim-visualizer` strip, review links) moved out of `Anotar` into a
  new full-width **`Frames`** view; viewer now has **six** tabs. `Anotar` keeps
  crop/classification/suggest/IMU/export plus an "Escolha de frames →" button.
- Large display: in `body.view-frames` the `.ssim-strip` becomes a wrapping grid
  (`repeat(auto-fill,minmax(340px,1fr))`) with frames up to `62vh` tall at
  `object-fit:contain` — every retained frame shown big and uncropped.
- Wiring: `'frames'` added to `VIEWS`; `setView('frames')` calls
  `loadSsimVisualizer()`; all moved controls keep their IDs so JS (incl. `Anotar`
  Export reading métrica/threshold) is unaffected. No duplicate IDs; template
  renders 200.
- Pages touched: [[viewer-frontend]].

## [2026-06-19] fix | video orientation reverted to measured (tag ignored)
- The 2026-06-18 tag-driven rule (`deg = (wantPortrait?90:0) - videoRotation`)
  laid almost every clip on its side. Ground-truth check: a decoded raw frame of
  the IFCE gimbal set (720×720, `rotate=90` tag) is **already upright**, and this
  browser does **not** honor the tag — so undoing it spun upright frames sideways.
- `index.html` `applyVideoOrientation()` now decides purely from the painted
  `videoWidth/videoHeight` vs `wantPortrait`: rotate 90° only on mismatch, leave
  square/already-matching frames at `deg = 0`. The container tag is no longer used
  for orientation (`_video_rotation()`/`DATA.video_rotation` still returned but
  unused). This restores the pre-`08d82c9` measured behaviour.
- Pages touched: [[viewer-frontend]].

## [2026-06-19] style | modern typography + visual refresh of the viewer
- `index.html` CSS: added design tokens to `:root` (colour palette, a type scale
  `--fs-xs…--fs-xl`, radius + shadow tokens); body now uses an Inter-first stack
  with antialiasing and 1.5 line-height.
- Then **scaled the whole UI up for readability** (sizes were too small): type
  scale bumped (base 15→16px, `--fs-xs` 11.5→13px … `--fs-xl` 22→26px) and the
  many hard-coded `px` font sizes converted onto the scale, so text flows from
  `:root`. Headline readouts enlarged (value numbers → `--fs-xl`, crop time →
  `--fs-lg`); left panel widened 480→540px and the three charts grew 155→190px.
- Modern polish: pill tab bar with accent-glow active state, rounded cards with
  soft shadows, `:focus-visible` accent rings, thin custom scrollbars, brighter
  `--text`/`--muted` for contrast. No HTML/JS structure changes.
- Verified: `/` renders 200; no remaining small hard-coded `px` font sizes.
- Pages touched: [[viewer-frontend]].

## [2026-06-19] refactor | remove dead suggestion code + split viewer into 5 tabs
- Backend cleanup (`app.py`): deleted the unused benchmark/template-matching
  walking detector (`_template_sustained`, `_walking_templates`,
  `_benchmark_profile`/`_benchmark_scores`/`_benchmark_sustained`,
  `_resample_features`, `_normalize_template`, `_walking_feature_series`,
  `_best_sustained`) and the orphaned `_lateral_acceleration_series` — 362 lines.
  Retired the `vertical` suggestion path: `suggest_crop` is walking-only
  (default `mode='walking'`), `api_suggest` default `vertical`→`walking` (kept
  `_first_sustained`, still used for horizontal walking-only clips).
- Frontend cleanup (`index.html`): dropped the "⬦ Vertical" button + listener,
  `runAutoSuggest` fallback now `['lateral','walking']`, trimmed `SUGGEST_*` maps;
  walking button relabelled "⬦ Caminhada".
- Viewer split into 5 client-side tabs (one route): Anotar | Inferência | Lote +
  Manifesto | Desvios + Validação | Datasets. `setView()` toggles a `view-*`
  body class (preserving orientation classes), persists to `localStorage`, and
  resizes charts/orientation on reveal. Video+charts player persists across the
  two video-centric tabs.
- Verified: import OK; `/api/suggest/0` walking & lateral → found; `?mode=vertical`
  → walking; `/` renders 200 with all 5 views balanced (5 open / 5 close).
- Pages touched: [[crop-suggestion]], [[viewer-frontend]], [[api-routes]].

## [2026-06-18] fix | video orientation is tag-driven (undoes container rotation)
- Backend `_video_rotation(idx)` reads each clip's display rotation tag (cv2
  `CAP_PROP_ORIENTATION_META`) and `get_clip_data` returns it as
  `video_rotation`. The viewer computes `deg = (wantPortrait ? 90 : 0) -
  videoRotation`, undoing the tag the browser already applied plus the
  supermarket portrait correction. `loadClip` sets `wantPortrait`/`videoRotation`
  (reset then updated from `/api/data`); also re-runs on `loadedmetadata`/resize.
- Supersedes the same-day measurement attempt, which skipped square frames: the
  IFCE gimbal set is 231 clips, all 720×720 (174 square / 57 portrait) with a
  90° tag, so dimensions can't orient them but the tag can. Verified
  `get_clip_data(0..3)` → video_rotation=90 → css deg=270 (−90°), upright.
- Pages touched: [[viewer-frontend]], [[api-routes]].

## [2026-06-18] change | viewer "Desvio lateral" button uses the PDF cut
- `/api/suggest?mode=lateral` now routes to `suggest_deviation_cut` (PDF decisão
  window T1-Δ→T1), matching the deviation export. Clips with no IMU desvio
  return `found:false`, so `runAutoSuggest` falls back to walking/vertical.
  `suggest_lateral_deviation` now backs only the batch/manifest lateral exports.
- Verified: `/api/suggest/0?mode=lateral` → found, [1.471, 2.471], window
  decisao, side LEFT, SSIM enrichment intact. Frontend unchanged (it reads
  start/end/ssim/has_vertical/message, not mode).
- Pages touched: [[crop-suggestion]], [[api-routes]].

## [2026-06-18] feat | deviation cut follows the criterios PDF (T1-Δ→T1)
- Added `suggest_deviation_cut(idx)`: takes the strongest `desvio` from
  `detect_imu_events` and returns a PDF §2.2 label window around T1/T2. Default
  `SENSECV_DEVIATION_CUT_WINDOW=decisao` → `[T1-Δ, T1]`, the "decisão visual"
  scene before the body reacts (PDF's recommended target for a predictive CNN;
  user-chosen). `acao` / `expandido` also available. Side comes from the same
  event so cut and label stay consistent.
- `export_set` gained `mode='deviation'`; the deviation export
  (`/api/inspect-deviation`) now cuts with this PDF window instead of
  `suggest_lateral_deviation`, and drops the old `_clip_deviation_side` helper.
- Verified: clip 01 t1=2.471 → cut [1.471, 2.471]; 6 clips exported with the
  full common-export folder content (mp4 + frames/accel/rotation json + ssim),
  validation.mp4 30 fps / 174 frames.
- Pages touched: [[crop-suggestion]], [[imu-event-labeling]], [[api-routes]].

## [2026-06-18] fix+rework | inspection video uses suggested cut + sensor side
- Reworked `/api/inspect-deviation` to drop the ML model entirely. Each clip now
  contributes **exactly its `suggest_lateral_deviation` cut**, labelled with the
  **sensor** deviation side from `detect_imu_events` (`_clip_deviation_side`:
  desvio direction esquerda/direita→LEFT/RIGHT, none→NONE). Title card + banner
  per cut; output cadence `play_fps` (default `SENSECV_INSPECT_FPS=15`).
- Fixed two latent bugs surfaced by the rework:
  - **`KeyError: '<clip name>'`** during long runs. `find_clips()` did
    `CLIP_PATHS.clear()` + repopulate; with `threaded=True`, the 5 s
    `/api/export-state` poll cleared the shared maps mid-rebuild while the
    inspection loop indexed them. Now `find_clips()` builds local dicts and
    **swaps the globals atomically** — readers see old or new, never empty.
  - **`_centered_rolling_mean` was undefined** (lost in restoration), so
    `detect_imu_events` / `/api/imu-events` raised `NameError`. Defined it as a
    list-tolerant centered moving average next to `_moving_average`.
- Verified (venv, TF present but unused): count=8 → 384 frames, 15 fps, 720×720;
  sides LEFT×3 RIGHT×5; `/api/imu-events/0` and `/api/sensemodel/0` still 200.
- Pages touched: [[api-routes]], [[imu-event-labeling]].

## [2026-06-18] feat | deviation inspection video (N clips or all)
- New `POST /api/inspect-deviation` `{count: N|"all", sample_fps?}` builds one
  stitched MP4 over the chosen clips with the active `.keras` model. Each clip
  is preceded by a title card naming its **dominant deviation**
  (`LEFT/RIGHT/NONE`, majority vote via `_dominant_label`) and every sampled
  frame carries the live `Desvio`/`Obstaculo` overlay (`_draw_label_bar`).
  Output `data/derived/deviation_inspection.mp4`; no `exports/`/history writes.
- Refactored the shared inference path into `_sensemodel_preprocess()` +
  `_split_head_arrays()` (reused by `sensemodel_frame_classification` and the
  new batched builder). Added `INSPECT_SAMPLE_FPS`/`SENSECV_INSPECT_FPS`.
- `templates/index.html`: new "Vídeo de inspeção (desvio)" panel with an
  "Inspecionar N" count input and "Todos" button (`runInspect`).
- Verified with the venv Python (TF 2.17): count=2 → 720×720, 4 fps, 69 frames,
  clip 01 NONE, clip 02 LEFT+OBSTACLE; output MP4 decodes in OpenCV.
- Pages touched: [[api-routes]].

## [2026-06-18] feat | upload any .keras model at runtime
- Made the SenseCV live model swappable without restarting Flask. The active
  path now lives in `_sensemodel_runtime['path']` (default still
  `SENSECV_MODEL_PATH` → `best_model.keras`); `_set_sensemodel_path()` resets
  the cached model and the per-frame prediction cache.
- New routes: `POST /api/upload-model` (saves to `data/models/`, activates and
  eagerly loads the upload), `POST /api/reset-model`, `GET /api/sensemodel-info`.
  Added `SENSECV_MODELS_DIR` (default `data/models`). `/api/sensemodel` now
  echoes the active `model` name.
- `templates/index.html`: SenseCV panel shows "Modelo ativo: <name>" and a
  **Trocar modelo / Padrão** upload row; `swapModel()` posts the file, relabels
  the panel, and re-runs inference on the current frame.
- Verified via Flask test client: info reports the default; a bogus upload saves
  with a sanitized name and returns an actionable load error (400); reset
  restores `best_model.keras`.
- Pages touched: [[api-routes]], [[dronet-live-classification]].

## [2026-06-16] fix | sensemodel preprocessing (raw 0-255, not /255)
- `/api/sensemodel` returned near-constant predictions. Cause: the exported
  model is MobileNetV2 with an embedded `Rescaling(1/127.5, -1)` layer, so it
  expects raw 0-255 input; the route was also dividing by 255, collapsing the
  effective range to ~[-1,-0.992]. Removed the `/255.0` — predictions now vary
  per frame (obstacle flips NONE/OBSTACLE, deviation LEFT/RIGHT/NONE). Color
  stays BGR→RGB, 224×224×3. Documented in [[api-routes]].

## [2026-06-16] frontend | wired SenseCV + manifest panels, enabled .keras model
- Added the **SenseCV (.keras) live panel** and **manifest export panel** to
  `templates/index.html`, the two UIs documented in [[viewer-frontend]] but
  missing from the recovered frontend. SenseCV inference piggybacks on
  `requestDronet()` so it shares the play/pause/scrub triggers; manifest export
  plans → exports clip-by-clip → builds the review MP4.
- Installed `tensorflow-cpu` into `.venv` and copied the user's
  `best_model.keras` (224×224×3, two heads 2+3) to the project root, so
  `/api/sensemodel` now returns real predictions instead of the controlled 503.
  Added `requirements-sensemodel.txt`.
- Deleted the stale `C:\Users\Isaque\Desktop\senseCV` copy (was junk + a subset
  of this checkout); its lone unique file was the lost-panel spec, now realized.

## [2026-06-15] recovery | rebuilt deleted wiki + re-implemented lost routes
- The `docs/wiki` tree was deleted; restored from the GitHub base plus replayed
  session edits, and the 11 DroNet/meta pages were copied from the intact
  `C:\Users\Isaque\sensecv` checkout. Both wiki copies are back to 38 pages.
- Recovered two dataset scripts missing from `sensecv` entirely
  (`make_clear_ssim_last1s_dataset.py`, `make_clear_metric_last1s_dataset.py`)
  from transcripts into `scripts/`.
- `/api/manifest-export` (+`/clip`, `/review`) and `/api/sensemodel` had no
  surviving source anywhere (only this wiki documented them), so they were
  **re-implemented from the spec**, not recovered: `plan_manifest_export()`,
  `export_manifest_clip()`, `build_manifest_review()`, and
  `sensemodel_frame_classification()` in `app.py`, mirroring the DroNet
  lazy-load + graceful-degrade pattern. Added `openpyxl` to `requirements.txt`.
- Verified by Flask test client: planning with a synthetic `.xlsx` queued 3
  clips with composed labels, a clip export cut + sensor/SSIM save succeeded,
  and the review MP4 built; `sensemodel` returns its controlled 503 (no `.keras`
  model in this checkout). Frontend wiring (batch manifest panel, sensemodel
  panel) is NOT yet added — backend + docs only.
- pages touched: [[api-routes]], [[log]]

## [2026-06-12] feature | IMU event labeling from the criterios PDF
- Implemented `data/uploaded_datasets/criterios_imu_video_orientando.pages.pdf`:
  `imu_quality_report()` (section 1 checklist + verdict), `detect_imu_events()`
  (desvio/reducao/parada with T1/T2, acao/decisao/expandido windows, and
  alta/baixa/descartar confidence), route `/api/imu-events/<idx>?delta=`,
  viewer "Eventos IMU" panel, and `scripts/imu_event_report.py` batch CSVs.
- Desvio uses gravity-projected yaw rate when the capture rotates, falling
  back to drift-corrected lateral velocity on gimbal-stabilized clips
  (max yaw ≈ 11°/s there). Validated on the 32 IFCE manifest clips: 26/32
  correct directions, 3 misses, 2 of 3 wrong flagged baixa.
- Pages touched: [[imu-event-labeling]] (new), [[api-routes]],
  [[viewer-frontend]], [[index]], [[log]].

## [2026-06-12] deploy | metric selector live locally and on Fly
- Local: `scripts/run_server.bat` on port 5000; all four metrics verified
  (`lpips`/`sewar` installed into `.venv`).
- Fly: redeployed `sensecv-api` (machine version 6). SSIM + VIF work; DINOv2 +
  LPIPS return a controlled "indisponivel" error (no torch in the slim image).
  Added `sewar` to `requirements.txt`.
- Fixed two deploy bugs: `[build].dockerfile` path in `deploy/fly.toml` is
  config-relative (`"Dockerfile"`), and `find_clips()` skipped the recursive
  scan when `CLIPS_DIR` == uploads root (`/data/clips`), which made the Fly app
  report `clips=0`.
- Pages touched: [[deployment-operations]], [[log]].

## [2026-06-12] feature | metric selector for frame selection in the viewer
- `app.py`: `ssim_frame_selection()` now takes `metric='ssim'|'dinov2'|'lpips'|'vif'`
  with lazy-loaded backends and per-metric default thresholds
  (`DEFAULT_METRIC_THRESHOLDS`); `/api/ssim` + `/api/suggest` accept `?metric=`,
  `/api/crop` accepts body `ssim_metric`. Score still reported as `ssim_prev`.
- `templates/index.html`: "Métrica" dropdown beside the threshold control;
  changing it resets the threshold to the metric default and widens the slider
  range for VIF (min 0.5). Suggest, visualizer, and export all send the metric.
- Verified via Flask test client: all 4 metrics return selections on clip 0;
  invalid metric → 400.
- Pages touched: [[api-routes]], [[viewer-frontend]], [[log]].

## [2026-06-12] ingest | DINOv2/LPIPS/VIF clear last-1s dataset variants
- Added `scripts/make_clear_metric_last1s_dataset.py`: same last-1-second
  selection pipeline as the SSIM clear supplement, with a pluggable metric
  (`--metric dinov2|lpips|vif`).
- Built all three over the same 112 uploaded clips (112/112 ok, labels match
  files): DINOv2 0.98 → 444 images, LPIPS 0.95 → 476, VIF 0.70 → 1386
  (SSIM 0.95 reference: 517). VIF's default threshold was lowered to 0.70
  because consecutive-frame VIF sits at 0.60–0.93.
- New deps installed: `lpips` 0.1.4, `sewar` 0.4.8 (DINOv2 via torch.hub).
- Variants are not merged into `data/derived/clear/images/`.
- Pages touched: [[clear-dataset]], [[index]].

## [2026-06-11] docs | synchronized wiki after clear repair and normalization
- Added [[clear-dataset]] to document the clear/no-obstacle image pool, the
  517-image SSIM 0.95 supplement, and the 12 stale label rows removed from
  `clear_ssim095_last1s_until_112`.
- Added [[dataset-normalization]] for the historical
  `normalized_14_730fps_420hz` build: 14.730 FPS video and 420 Hz sensor
  streams.
- Updated backend/API/frontend wiki pages for current `data/` paths, recursive
  dataset discovery, capture-quality badges, `/api/sensemodel`, and local
  model/runtime notes.
- Pages touched: [[clear-dataset]], [[dataset-normalization]], [[api-routes]],
  [[viewer-frontend]], [[app-backend]], [[index]], [[log]].

## [2026-06-09] git | published source-only project commit to GitHub
- Read the wiki and followed the [[github-repository]] source-only policy:
  local datasets, exports, `history.json`, `.env`, logs, generated media, and
  Python caches remain ignored.
- Added `.env` to `.gitignore` while keeping `.env.example` tracked.
- Created local commit `99dd97a Commit SenseCV project source`, then found the
  checkout had no `origin` remote configured, so the commit was not visible on
  GitHub.
- Added `origin https://github.com/IsaqueEfraimIFCE/sensecv.git`.
- Initial push was rejected because GitHub already had remote history through
  `bcf1859 Load Fly example dataset`; histories were unrelated.
- Preserved the local root commit as branch `local-root-snapshot`, created a
  publish commit on top of `origin/master`, and pushed a fast-forward update:
  `bcf1859..8d3c6ad`.
- Current GitHub `master` is
  `8d3c6ad4ece95653f3dbd97d78ac4d6301bc9042`
  (`Publish SenseCV project source`), and local `master` tracks
  `origin/master`.
- Verification: `python -m py_compile` passed for `src/sensecv/app.py` and
  the scripts using an ignored temporary cache; remote `refs/heads/master`
  resolved to `8d3c6ad4ece95653f3dbd97d78ac4d6301bc9042`.
- Pages touched: [[github-repository]], [[log]].

## [2026-06-06] merge | unified DroNet source/wiki into PilotGuru
- Vendored the durable DroNet source layer from `C:\Users\Isaque\Desktop\dronet`
  into `dronet/`: PyTorch model port, helper scripts, paper sources, and
  upstream `repo/` including `model/model_weights.h5`.
- Updated `app.py`, `run_samsung_dronet_samples.py`, and
  `run_sensecv_02062026_dronet_3fps.py` so `DRONET_DIR` defaults to the local
  `dronet/` folder while still allowing environment overrides.
- Updated `dronet/run_dronet.py` to use local `exports/` and `dronet_results/`
  defaults instead of absolute Desktop paths.
- Merged the standalone DroNet wiki into this canonical wiki by adding
  [[dronet-overview]], [[source-inventory]], [[dronet-paper]],
  [[model-architecture]], [[preprocessing]], [[inference-pipeline]],
  [[experiment-results]], [[pilotguru-10fps-folder-check]], [[ros-control]],
  and [[open-questions]] to [[index]].
- Pages touched: [[overview]], [[index]], [[log]], [[dronet-overview]],
  [[source-inventory]], [[open-questions]].

## [2026-06-03] deploy | loaded 02-06-2026 IFCE example dataset on Fly
- Uploaded the local `SenseCV-02-06-2026-IFCE-Gimbal` dataset to the Fly
  volume through `/api/upload-zip` as the example dataset.
- Fixed duplicate clip discovery when `SENSECV_EXTRA_CLIP_ROOTS` and
  `SENSECV_UPLOADS_DIR` point at the same `/data/clips` root. Uploaded
  recursive datasets now display as `SenseCV-.../01` instead of
  `clips/SenseCV-.../01`.
- Deployed new image:
  `sensecv-api:deployment-01KT788C7KJBMCBXJMYGPPXCJE`.
- Verified `https://sensecv-api.fly.dev/health` returned HTTP 200 with
  `{status:"ok", clips:32}`.
- Verified `/api/clips` starts with
  `SenseCV-02-06-2026-IFCE-Gimbal/01`; `/api/data/0` returns 202 frame
  samples and 57 active external input frames.
- Pages touched: [[deployment-operations]], [[log]].

## [2026-06-03] fix | Fly no-dataset startup and warm machine
- Fixed the viewer boot path for source-only Fly deployments with zero clips.
  `templates/index.html` now renders a no-dataset state and keeps the zip
  import controls available instead of calling `loadClip(0)`.
- Updated `fly.toml` to `min_machines_running = 1` so `sensecv-api` keeps one
  machine warm during interactive testing.
- Deployed new image:
  `sensecv-api:deployment-01KT77HJCXDTV5X7V7BKWNN456`.
- Verified machine `0807567b062618` is version `3`, state `started`;
  `/health` returns HTTP 200 with `{status:"ok", clips:0}`; the served root
  HTML includes `showNoClipsState`.
- Pages touched: [[deployment-operations]], [[log]].

## [2026-06-03] deploy | updated Fly app to latest source image
- Deployed `sensecv-api` to Fly.io with `flyctl deploy -a sensecv-api`.
- New image:
  `sensecv-api:deployment-01KT76K9AY62E3FT637KJ9N7SV`.
- Machine `0807567b062618` is version `2` and reached `started` state.
- Verified `https://sensecv-api.fly.dev/health` returned HTTP 200 with
  `{status:"ok", clips:0}`. This is expected for the source-only image before
  importing datasets onto `/data/clips`.
- Verified `/api/clips` returned an empty clip list and `/api/upload-zip`
  returned the controlled missing-zip error.
- Pages touched: [[deployment-operations]], [[log]].

## [2026-06-03] docs | GitHub remote and current SenseCV project state
- Captured the GitHub project state in [[github-repository]]: local branch
  `master`, remote `origin https://github.com/IsaqueEfraimIFCE/sensecv.git`,
  GitHub account `IsaqueEfraimIFCE`, and the latest committed hashes known at
  the time of documentation.
- Documented the source-only policy for the GitHub repository: datasets, MP4s,
  generated DroNet outputs, upload folders, exports, and local runtime files
  stay out of Git through `.gitignore` / `.dockerignore`.
- Normalized Fly deployment docs to the real lowercase app name
  `sensecv-api` and public URL `https://sensecv-api.fly.dev/`.
- Pages touched: [[overview]], [[index]], [[github-repository]],
  [[deployment-operations]], [[api-routes]], [[log]].

## [2026-06-03] feature | live DroNet classification in Flask viewer
- Added `/api/dronet/<idx>?time=<seconds>&exact=<0|1>` to `app.py`.
  `exact=1` classifies the nearest paused frame; `exact=0` buckets time to
  3 FPS for playback updates.
- Backend lazily loads the local Desktop DroNet PyTorch port and H5 weights
  on first inference request, then caches runtime state and up to 300
  frame-level predictions keyed by clip, video mtime, and frame index.
- Added a compact DroNet panel to `templates/index.html` under the live sensor
  values. It shows direction/steering, yaw degrees, collision probability, and
  the source frame/time. Collision probability >= 0.5 gets red emphasis.
- Frontend behavior: initial clip load and paused/scrubbed frames request exact
  inference; playback requests are deduped to one request per 3 FPS bucket.
- Verified `python -m py_compile app.py`, Flask test client for
  `/api/dronet/0?time=0&exact=1`, and live server on
  `http://127.0.0.1:5001` because port 5000 had a stale/nonmatching process.
- Pages touched: [[overview]], [[index]], [[app-backend]], [[api-routes]],
  [[viewer-frontend]], [[dronet-live-classification]],
  [[dronet-sensecv-02062026-3fps]], [[log]].

## [2026-06-03] ingest | DroNet 3 FPS classifications for SenseCV 02-06-2026 gimbal
- Added `run_sensecv_02062026_dronet_3fps.py`, which imports the local DroNet
  model from `PilotGuru\dronet`, samples every
  `SenseCV-02-06-2026-IFCE-Gimbal` clip at 3 FPS, and writes annotated PNG
  classifications.
- Latest run processed 32 clip folders and wrote 666 frame classifications to
  `dronet_sensecv_02062026_3fps/`.
- Each output subfolder contains `classifications.csv`, annotated
  `classification_*.png` images, and a `contact_sheet.png` with all generated
  images for that clip.
- Aggregate outputs are `all_classifications.csv` and `summary.json`.
- Highest mean collision clips were `21`, `25`, `26`, `22`, `32`, `23`, `24`,
  and `31`; these remain DroNet inspection scores, not validated indoor labels.
- Pages touched: [[overview]], [[index]], [[dronet-sensecv-02062026-3fps]],
  [[log]].

## [2026-06-03] feature | GitHub-ready SenseCV app, zip dataset import, Fly rename
- Added GitHub-facing project files: `README.md`, `.gitignore`,
  `requirements.txt`, `Dockerfile`, `fly.toml`, and `.dockerignore`.
- Renamed the application identity to SenseCV in app configuration, Fly app
  naming, and wiki prose.
- Added `/api/upload-zip` and the "Importar dataset SenseCV" UI module for
  importing zip archives whose nested folders match the SenseCV clip schema.
- Zip import safely rejects path traversal, validates that at least one valid
  clip folder exists, extracts into `SENSECV_UPLOADS_DIR`, refreshes discovery,
  and returns updated clip/group lists.
- Deployed the renamed Fly app `sensecv-api`, verified `/health`, `/api/clips`,
  `/api/data/0`, and the upload endpoint's controlled empty-upload error.
- Pages touched: [[overview]], [[index]], [[app-backend]], [[api-routes]],
  [[viewer-frontend]], [[deployment-operations]], [[log]].

## [2026-05-31] ingest | Samsung/Poco 30-05-2026 gimbal roots and Samsung DroNet samples
- `app.py`: added the inner Downloads roots for
  `SenseCV-30-05-2026-IFCE-Gimbal-Samsung` and
  `SenseCV-30-05-2026-IFCE-Gimbal-Poco` to `EXTRA_CLIP_ROOTS`.
- Verified app discovery at 305 clips total: 48 Samsung clips and 54 Poco clips
  from the new roots. Both join `WALKING_ONLY` through the existing extra-root
  path.
- Read `Coleta  IFCE - 30 de maio de 2026 - ANDRÉ (Samsung) - Página1.csv`.
  It has 53 physical rows but 48 non-empty unique IDs (`1`-`48`), matching the
  Samsung clip folders.
- Added `run_samsung_dronet_samples.py`, which imports the local DroNet model
  from `PilotGuru\dronet`, samples 20 deterministic random frames
  per Samsung ID folder, and writes annotated PNGs plus CSV/JSON summaries.
- Latest run wrote 960 predictions to `dronet_samsung_random_samples/`; all 48
  Samsung clips decoded successfully.
- Pages touched: [[overview]], [[index]], [[app-backend]], [[SenseCV-dataset]],
  [[dronet-samsung-samples]], [[log]].

## [2026-05-30] feature | export-aware clip selection, clean export MP4s, live state
- `app.py`: clip discovery now includes `exports/` as selectable clips. Raw
  clips use `video.mp4`; exports use `<folder>.mp4`, so `CLIP_VIDEO_PATHS`
  stores the exact MP4 path. `CLIP_GROUPS` drives source grouping in the UI.
- `templates/index.html`: added a folder/group selector that jumps directly to
  the first clip in a group, while the clip selector uses optgroups. Groups now
  include `Supermercado Telefrango`, each SenseCV root, and `exports`.
- Auto-suggest now tries `lateral` first, then `walking`, then `vertical`.
- `history.json` is pruned against actual export folders on load. Next export
  number comes from the first missing leading number in `exports/`; no numbered
  exports means `01`.
- Added `/api/export-state` polling to refresh clips, groups, history, export
  folders, and next number without restarting.
- `run_server.bat` / `run_server.ps1`: stop existing port-5000 listeners before
  launching this checkout's `.venv` Python and `app.py`, preventing stale Flask
  processes from serving old templates.
- Export video cutting now re-encodes with H.264 (`libx264`) instead of
  `-c copy`, preserving rotation metadata and fixing short/keyframe-broken
  exported clips. Repaired 24 history-backed exports; two folders without
  history records (`06_obstaculo_esquerda_desvio_direita`, `erro4`) were left
  unchanged.
- Pages touched: [[overview]], [[index]], [[app-backend]], [[viewer-frontend]],
  [[api-routes]], [[clip-data-model]], [[exports-output]], [[history-json]],
  [[crop-suggestion]], [[export-pipeline]], [[log]].

## [2026-05-28] ingest | new SenseCV source root: SenseCV-27-05-2026-IFCE-Gimbal (24 clips)
- Added `C:\Users\Isaque\Desktop\SenseCV-27-05-2026-IFCE-Gimbal` as a 4th entry
  in `EXTRA_CLIP_ROOTS` (`app.py:21`). Gimbal-stabilized variant of the
  27-05-2026 footage.
- All 24 subfolders (`01`–`24`) validated as clip dirs per `_is_clip_dir`
  (each has `video.mp4` + `frames.json` + the usual sensor/extra JSONs).
- Surfaces in the UI with the `SenseCV-27-05-2026-IFCE-Gimbal/NN` display
  prefix and joins `WALKING_ONLY` (horizontal footage — vertical-orientation
  detection skipped, per [[orientation-detection]]).
- Total clip count after restart: **281** (was 257; +24). Verified via
  `/api/clips`.
- pages touched: [[log]]

## [2026-05-28] ingest | new SenseCV source root: SenseCV-27-05-2026-IFCE (33 clips)
- Extracted `SenseCV-27-05-2026-IFCE.zip` to
  `C:\Users\Isaque\Downloads\SenseCV-27-05-2026-IFCE\` (single level; zip
  already wrapped the folder, unlike the 2023 root's double-nested path).
- All 33 subfolders (`01`–`33`-ish) validated as clip dirs per `_is_clip_dir`
  in `app.py` (each has `video.mp4` + `frames.json` + `accelerations.json` +
  `rotations.json`).
- Wired into `EXTRA_CLIP_ROOTS` in `app.py:17-20`; clips will surface in the
  UI with the `SenseCV-27-05-2026-IFCE/NN` display prefix and join
  `WALKING_ONLY` (horizontal footage — vertical-orientation detection
  skipped, per [[orientation-detection]]).
- Total clip count: 157 supermercado + 67 (23-05-2023) + 36 (26-05-2026) +
  33 (27-05-2026) = **293** (was 260 before this ingest).
- Each clip also ships `gps_status.json`, `locations.json`,
  `magnetometer.json`, and a stub `external_sensors.json` — none of these are
  consumed by the current pipeline (only video / frames / accel / rotations
  are read), so [[clip-data-model]] still reflects the system's view; the
  extras are noise to it.
- Pages touched: [[log]] (this entry); [[app-backend]] unchanged in behavior
  but its `EXTRA_CLIP_ROOTS` list now has a third entry.

## [2026-05-28] redesign | lateral detector now scores mean |lateral velocity|
- Per user request, swapped the lateral detector from yaw rate to mean
  |lateral velocity| over a sliding 1 s window. Lateral velocity = integrated
  horizontal accel, decomposed against the local rolling-mean walking
  direction; the perpendicular component is the lateral one.
- Benchmark against the same 5 GT clips: lateral-velocity scored 0.61 s mean
  abs offset, slightly better than yaw rate's 0.68 s and comparable to a
  z(yaw)+z(lat-vel) combination (0.53 s — kept out for simplicity).
- Pages touched: [[crop-suggestion]], [[log]].

## [2026-05-28] redesign | lateral detector switched from accel-mag to yaw rate
- Calibrated `suggest_lateral_deviation()` against the operator's 5 deviation
  exports in [[history-json]] (entries 6-10, SenseCV clips 28/29/30/31/46).
- Debug script `_debug_lateral.py` showed lateral-accel magnitude is a poor
  feature: for 3 of 5 GT clips the global lateral-mag max lay outside the
  labeled window, and a sliding 1 s mean over it also picked the wrong window.
  Yaw rate (gyro component along the gravity axis = body pivoting) matched
  all 5 GT windows within ~0.5 s.
- Rewrote the detector around a sliding 1 s window scored by mean yaw rate,
  always returning a window (per the user's "must happen in every lateral
  deviation clip" requirement). Best match: clip 197 within 0.03 s; mean
  absolute start offset 0.30 s, end offset 0.38 s.
- `SenseCV dataset (lateral)/` is now stale again — re-run the lateral
  batches via the UI to refresh.
- Pages touched: [[crop-suggestion]], [[log]].

## [2026-05-27] tune | lateral detector: score by peak, not length
- `suggest_lateral_deviation()` now scores candidate runs by **peak magnitude**
  instead of duration, raises the bar to a two-tier threshold (sustain 1.5 +
  peak 3.5 m/s²), and trims the kept run to a ≤ 1.2 s window centered on the
  peak. Reasoning: an obstacle-avoidance sidestep is a sharp, brief impulse;
  the prior "longest above 0.8 m/s²" rule kept slowly drifting low-amplitude
  windows instead.
- Reduced pre-smoothing in `_lateral_acceleration_series()` from ~0.3 s to
  ~0.1 s so the impulse peak isn't flattened out before scoring.
- Spot-checks: supermercado clip 0 went from 1.6–12.4 s (11 s of mild motion)
  to a 1.1 s window centered on its true peak at 25.9 s. SenseCV walking-only
  clips that previously matched the loose threshold now correctly report no
  detection.
- The previous `SenseCV dataset (lateral)/` run (102 ok) is now stale; re-run
  the SenseCV / Supermercado lateral batches via the UI to refresh.
- Pages touched: [[crop-suggestion]], [[log]].

## [2026-05-27] feature | UI batch export + lateral-deviation cut mode
- `app.py`: added `suggest_lateral_deviation()` — single longest sustained
  horizontal-accel burst (smoothed 0.3 s, threshold 0.8 m/s², minimum 0.2 s),
  with gravity removed via 1 s rolling-mean so it works in any phone
  orientation. Wired through `/api/suggest?mode=lateral`.
- `app.py`: moved `export_set()` + preset helpers into the module so the
  Flask process and the CLI share one code path (avoids the circular-import
  trap that `from export_SenseCV import export_set` would have triggered).
  Added `POST /api/batch-export` (synchronous; returns ok/skipped/failed).
- `templates/index.html`: third "⬦ Desvio lateral" suggest button; new
  "Exportação em lote" panel with 4 preset×mode buttons; generalized
  `runSuggest` via `SUGGEST_BTN_IDS` / `SUGGEST_LABELS` maps.
- `export_SenseCV.py`: collapsed into a thin CLI around `app.export_set`,
  accepts `[preset] [mode]` args. Lateral output goes to
  `<base> dataset (lateral)/` to avoid clashing with walking cuts.
- Verified end-to-end via CLI `SenseCV lateral`: 102 ok, 1 below-threshold
  skip, 0 failed.
- Pages touched: [[crop-suggestion]], [[api-routes]], [[viewer-frontend]],
  [[app-backend]], [[SenseCV-dataset]].

## [2026-05-27] feature | Supermercado dataset batch export (sibling to SenseCV)
- Refactored `export_SenseCV.py` around `export_set(indices, out_dir)` driven by
  two `PRESETS` (`SenseCV` default, `supermarket` opt-in via argv). The
  supermarket preset selects every clip *not* in `WALKING_ONLY` and writes to
  `Supermercado dataset/`, with its own `sources.csv`.
- Calls `suggest_crop(idx, 'walking')` explicitly so supermercado clips go
  through the learned-window / Random Forest path ([[crop-suggestion]]) rather
  than the broader vertical mode.
- Last run: 71 ok, 86 no-walking, 0 failed across the 157 supermercado clips —
  same coverage as the UI's auto-suggest.
- Also fixed a Windows cp1252 stdout crash from a `→` literal in the progress
  print.
- Pages touched: [[SenseCV-dataset]] (broadened to cover both datasets),
  [[index]], [[log]].

## [2026-05-27] feature | SenseCV dataset batch export + provenance CSV
- Added `export_SenseCV.py`: iterates every clip in `WALKING_ONLY`, runs the
  horizontal-branch `suggest_crop()` ([[crop-suggestion]]), ffmpeg-cuts the
  walking segment with `-c copy`, and reuses `save_sensor_data()` for the
  rebased per-frame/sensor JSON ([[export-pipeline]]).
- Outputs land in `SenseCV dataset/<source-prefixed-name>/` (e.g.
  `SenseCV-23-05-2023-IFCE_06/`), with a `sources.csv` ledger mapping each
  output back to its source root, subfolder, absolute path, and crop window.
- Deliberately bypasses `history.json` so SenseCV cuts do not enter the walking
  classifier's training set — keeps the horizontal regime out of the
  supermarket suggestion loop. See [[SenseCV-dataset]].
- Pages touched: [[SenseCV-dataset]] (new), [[index]], [[log]].

## [2026-05-27] feature | multi-root clip input + horizontal walking-only clips
- `app.py`: clip discovery now scans `CLIPS_DIR` plus `EXTRA_CLIP_ROOTS`
  (two `SenseCV-*` folders). Added `CLIP_PATHS` (name→abs path) and
  `clip_path()` resolves through it; extra-root clips get a `label/name`
  display name to stay unique (their subfolders are just `01`,`02`,…). Primary
  root clips keep bare names, so existing history/exports stay valid.
- `app.py`: added `WALKING_ONLY` set. `suggest_crop()` branches early for those
  clips — ignores the vertical mask and segments the first sustained walking run
  (`_first_sustained(walking,…)`), honoring learned windows; returns
  `has_vertical:true` so the warning never fires. Only the supermarket (primary)
  root still uses vertical-orientation detection.
- Clip count 157 → 260 (+67 +36). Verified: SenseCV clip 157 → walking 0.0–9.07,
  supermarket clip 0 → learned 18.999–25.62.
- Pages touched: [[crop-suggestion]], [[orientation-detection]], [[app-backend]],
  [[clip-data-model]].

## [2026-05-26] feature | learned walking crop suggestions from exports
- `app.py`: added `learned_walking_window()` so walking suggestions use the
  latest exported crop for a source clip before falling back to sensor
  heuristics; added `_best_sustained()` for new clips without export history.
- `templates/index.html`: auto-suggest now requests `walking` mode instead of
  the broader vertical-only mode.
- Verified the five current `history.json` exports match walking suggestions
  exactly.
- Pages touched: [[crop-suggestion]], [[app-backend]], [[viewer-frontend]].

## [2026-05-26] feature | benchmark-based walking fallback
- `app.py`: replaced the generic walking fallback with a benchmark scorer built
  from exported crop windows and nearby non-crop context; unexported clips now
  use `_benchmark_profile()` + `_benchmark_sustained()`.
- Hardened suggestion handling for short/malformed clip sensor files so the API
  returns a controlled no-suggestion response instead of raising.
- Verified all 157 clips: 95 suggestions, 62 controlled no-suggestion responses,
  0 crashes.
- Pages touched: [[crop-suggestion]], [[app-backend]].

## [2026-05-26] feature | template-matched single walking window
- `app.py`: changed unexported walking suggestions from frame scoring to sliding
  sequence matching against the exported ground-truth windows. The fallback now
  returns one candidate window per clip: the segment whose smoothed sensor
  sequence is closest to the exported walking templates.
- Added a `history.json` mtime cache for the walking templates so new exports
  refresh the benchmark automatically.
- Verified all 157 clips: 111 suggestions, 46 controlled no-suggestion
  responses, 0 crashes.
- Pages touched: [[crop-suggestion]], [[app-backend]].

## [2026-05-26] feature | IMU/Gyroscope walking detector
- `app.py`: replaced the active walking fallback with `_imu_walking_window()`,
  which predicts walking from accelerometer high-pass energy, gyroscope energy,
  jerk, and portrait gating.
- Calibrated against the five exported ground-truth windows; learned exports
  still return exact saved windows, while unexported clips use the IMU detector.
- Full scan: 157 clips, 88 walking suggestions, 69 controlled no-suggestion
  responses, 0 invalid-duration suggestions.
- Pages touched: [[crop-suggestion]], [[app-backend]].

## [2026-05-26] feature | supervised walking classifier
- `app.py`: added a cached Random Forest frame classifier trained from
  `history.json` exports, using IMU/gyroscope energy plus rolling gait-cadence
  autocorrelation features.
- `suggest_crop(..., mode='walking')`: unexported clips now use the classifier
  fallback, then probability smoothing and one-window segment selection.
- Validation: exact exported clips still return saved windows; classifier-only
  benchmark error on the 5 labeled clips is below 0.25 s per clip; full scan
  found 71 valid windows, 86 controlled no-suggestion responses, 0 bad
  durations/crashes.
- Pages touched: [[crop-suggestion]], [[app-backend]].

## [2026-05-26] feature | auto-suggest, vertical warning, synced playback, walking mode
- Code change ingested from a session that modified `app.py` + `templates/index.html`.
- `app.py`: refactored `suggest_crop` into `_orientation_walking_masks()` +
  `_first_sustained()`; added `mode` param (`vertical`/`walking`) and
  `has_vertical` to the response; `/api/suggest` now reads `?mode=`.
- `templates/index.html`: auto-suggest on clip load, "no vertical moment"
  warning banner, play/pause button + `requestAnimationFrame` sync loop, second
  "Vertical + andando" suggest button via `runSuggest(mode)` with stale guard.
- Verified on real clips: clip0 vertical 14.23→25.62, walking 22.39→25.62;
  clip2 has_vertical=false (warning case).

## [2026-05-26] init | wiki created
- Bootstrapped the LLM Wiki for SenseCV from the codebase (`app.py`,
  `templates/index.html`) as raw source.
- Wrote schema (`CLAUDE.md`), [[index]], [[overview]], and 13 entity/concept
  pages covering the backend, frontend, data model, API, and the sensor /
  crop / export algorithms.
- Pages touched: all (initial creation). Page count: 16.
- Open questions filed: whether [[walking-detection]] should use fused `speed`
  from [[velocity-estimation]]; frame-exact vs keyframe-copy cut tradeoff in
  [[export-pipeline]].

## [2026-07-21] feature | compass turn labeling selector
- `app.py`: added denoised, tilt-compensated magnetometer headings sampled at
  10 Hz, wraparound-safe 0.5 s deltas, the ±5 degree LEFT/RIGHT/FRONT rule, and
  `GET /api/compass-turn/<idx>`; exported cuts retain rebased magnetometer data.
- `templates/index.html`: added Bússola, Normal, and Nenhum direction-selection
  modes; FRONT or no automation leaves direction unset and prevents an
  incomplete deviation export.
- Verified Python compilation, rendered JavaScript syntax, the Flask route on a
  real clip, and the 358 to 3 degree wraparound case.
- Pages touched: [[compass-turn-labeling]], [[viewer-frontend]], [[api-routes]],
  [[app-backend]], [[export-pipeline]], [[index]].

## [2026-07-21] change | compass-only turn selector
- Removed the Normal and Nenhum method buttons from the viewer; Bússola is now
  the sole and default automatic direction method, with manual correction still
  available through the direction chips.
- Pages touched: [[compass-turn-labeling]], [[viewer-frontend]], [[index]].

## [2026-07-27] change | exclude compound-CAIXA clips from DINOv2 endpoints
- `train_dinov2_endpoints_kaggle.py` now excludes the 44 clips whose label
  manifest `OBSTACULO` value is `CAIXA` plus a qualifier before choosing each
  sequence's first and last frames; a plain `CAIXA` remains eligible.
- The startup summary and `metrics.json` report the excluded clip/frame counts
  and `metrics.json` preserves the exact excluded `source_clip` list.
- Verified against the local obstacle/free-path `labels.csv`: 2,066 source
  frames from 44 clips excluded and 1,590 endpoint frames retained.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-27] change | train DINOv2 linear probe on all eligible frames
- `train_dinov2_endpoints_kaggle.py` now retains every frame from each eligible
  clip instead of selecting only the first and last frame per sequence.
- Compound-CAIXA exclusion and clip-grouped train/validation/fixed-day test
  splitting remain in place, preventing frame leakage between splits.
- The output directory and `metrics.json` selection name now identify the
  all-frame run; the older endpoint dataset saver and MobileNet comparison
  remain endpoint-only.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-27] feature | label-safe left/right flip augmentation
- The all-frame DINOv2 trainer now adds one horizontally flipped copy of every
  training image for left/right datasets, swapping `left` and `right` labels
  while preserving `none`.
- Validation/test are unchanged, obstacle/free-path skips the augmentation, and
  both the checkpoint and metrics report source/flipped/effective frame counts.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-27] dataset | unique reviewed exports from UFC 24-07
- Added `make_kaggle_exports_dataset.py`: it recovers export provenance by exact
  accelerometer matching, validates labels against the collection XLSX, and
  writes 224×224 `obstacle_freepath` / `deviation_lrn` ImageFolder datasets.
- Built 27 unique raw clips and 814 images per scheme from the reviewed exports
  only; unused raw IDs 03/13/28 contribute nothing.
- First appearance wins: later exports 05/08/13/18 were ignored because their
  raw clips already appeared in exports 04/07/12/17.
- Verified all label paths, 224×224 RGB samples, class/source counts, ZIP CRC,
  and archive SHA-256.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-27] training | add UFC 24-07 exports to DINOv2 train split
- `train_dinov2_endpoints_kaggle.py` auto-discovers the unique reviewed-export
  package and appends its matching task rows to training only, after the base
  validation/test split is fixed.
- The attachment is required by default; label mismatches fail, and clips that
  overlap the base dataset are skipped to avoid duplicate or cross-split data.
- Verified obstacle training: 17,199 base + 814 supplemental = 18,013 frames.
  Verified deviation training: 16,579 base + 814 supplemental = 17,393 source
  frames, then label-safe flips produce 34,786 effective frames. Validation and
  test counts are unchanged.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-27] training | hold out both day 22 and July 24
- The DINOv2 all-frame trainer now merges the reviewed July-24 package before
  splitting and uses `test_days=[22,24]`.
- Test is 118 clips / 3,970 frames: the original 3,156 day-22 frames plus all
  814 July-24 frames. No test clip overlaps train/validation or receives flip
  augmentation.
- Obstacle test labels are 2,230 obstacle / 1,740 free-path; deviation test
  labels are 1,247 left / 983 right / 1,740 none.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-27] training | add only new auto-trim dates to train/validation
- The DINOv2 trainer now materializes all frames from
  `Todos_cortes_auto_trim/deviation` into a task-specific dataset under its
  output directory, then includes those rows in the train/validation split.
- Of 567 source folders, 171 new clips (29,124 frames) from July 2/3/5/6/7 are
  accepted; 352 already-loaded clips and all 44 CAIXA clips are ignored.
- Full dates and `source_clip` identities are compared against both existing
  datasets, preventing duplicate dates/clips. Days 22 and 24 remain test-only.
- The built dataset writes class folders, `labels.csv`, and
  `dataset_info.json`; checkpoint/metrics include its provenance and skip
  counts.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-27] training | balance background frames within clips
- The all-frame DINOv2 trainer now downsamples `free_path` to the obstacle
  count, or `none` to the combined left/right count, independently in train
  and validation while preserving every foreground frame.
- Exact quotas are distributed proportionally across background clips, and
  evenly spaced frames are retained within each clip. Days 22 and 24 keep
  their natural test distribution.
- Checkpoint and metrics output record pre/post totals and per-clip keep
  fractions; left/right flip augmentation remains training-only and runs
  after balancing.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-27] training | do not flip none frames
- Horizontal augmentation now creates copies only for `left` and `right`,
  swapping their labels. The balanced `none` frames are retained only in their
  original orientation.
- Validation and test remain unaugmented; augmentation metrics now explicitly
  record `none` as an excluded label.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-27] training | exclude collection day 9
- `CONFIG["excluded_days"] = [9]` now removes all day-9 rows before splitting,
  so they cannot enter train, validation, or test.
- Day-9 auto-trim folders are skipped before video decoding. Checkpoint,
  metrics, and console output record the excluded clips and frame count.
- Days 22 and 24 remain the fixed test holdout.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-27] training | replace linear probe with three-layer MLP
- The frozen DINOv2 embeddings now feed dense layers `384 -> 256 -> 64 ->
  classes`, with ReLU activations and dropout rates 0.3/0.2 between layers.
- Hidden dimensions and dropout are configurable. The resolved architecture is
  printed and saved in both the checkpoint and metrics.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-28] training | restore linear probe and equalize deviation validation
- The classifier is restored to one dense layer from the frozen DINOv2
  embedding directly to the class logits; the temporary 256/64-neuron MLP and
  dropout configuration were removed.
- Deviation validation now clip-proportionally samples every class to its
  smallest class count, producing equal `none`, `left`, and `right` totals
  without augmenting validation.
- Direction-only training flips still never copy `none`; together with
  pre-flip background balancing they produce equal effective train totals for
  `none`, `left`, and `right`; a final equal-class sampling pass guarantees
  this even if a future input has fewer `none` frames than directional frames.
  Obstacle/free-path balancing is unchanged.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-28] training | balance only none and split by balanced frame totals
- Removed the temporary all-class equalizer: `left` and `right` are no longer
  downsampled to match each other in train or validation.
- For deviation validation and test, only `none` is sampled, targeting the
  rounded mean of the unchanged left/right frame counts. Training retains its
  prior pre-flip background target and direction-only flips never copy `none`.
- The train/validation candidate pool is balanced before clip selection.
  A nearest whole-clip subset search chooses validation clips by balanced frame
  totals, rather than choosing 15% of clips regardless of clip length.
  Deviation compensates its selection target for the different train/validation
  `none` policies so the final post-balance validation share approaches 0.15;
  original rows are recovered before per-split sampling.
- Obstacle/free-path test data remains naturally distributed.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-28] training | restore independent train/validation background balance
- Restored the earlier trainer policy: clip-count-based grouped splitting,
  followed by independent train and validation background sampling.
- `free_path` matches obstacle frames; `none` matches combined left/right
  frames. Foreground rows are preserved and background quotas are distributed
  proportionally across clips with evenly spaced temporal selection.
- Test days 22 and 24 retain their natural distribution. Later day-9
  exclusion, mean-of-directions sampling, frame-target split search, and
  all-class balancing were removed.
- Restored that version's augmentation behavior: the balanced training set is
  duplicated, flips swap left/right, and flipped `none` remains `none`.
- The classifier remains the original single linear layer.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-28] training | remove balancing and retain day-9 exclusion
- Removed all train/validation background sampling: every eligible frame is
  retained after the clip-grouped split, with no class or frame-rate balance.
- Restored `excluded_days=[9]` for packaged rows and auto-trim folders; auto
  day-9 videos are skipped before decoding.
- Days 22 and 24 retain their natural test distribution. Training-only flip
  augmentation and the single linear classifier are unchanged.
- Removed background-balance fields from checkpoint and metrics; day-9
  exclusion provenance remains recorded.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-28] training | remove undated-clear day-22 fallback
- `read_training_rows()` no longer rewrites undated `sem_obstaculos` or
  `source=clear` rows as day 22.
- Test-day membership now depends only on an explicit `DD-MM-YYYY` in
  `source_clip`. Existing packaged clear rows with dated day-22 metadata remain
  day-22 test frames.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-28] dataset | add all-frame train_val/test materializer
- Added `save_dinov2_allframes_trainval_test_dataset.py`, which reuses the
  trainer's ingestion declarations without running training.
- Output has only `train_val/<class>` and `test/<class>`; no train/validation
  split, balancing, subsampling, or augmentation is applied.
- Day 9 and compound-CAIXA clips are excluded; explicitly dated days 22 and 24
  form test. Base, reviewed July-24, and eligible auto-trim frames are included.
- Root `labels.csv` and `split_summary.json` preserve split and source
  provenance. Auto-trim decoding uses a temporary cache removed after copying.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-28] dataset | make train_val/test materializer standalone
- Inlined dataset discovery, compound-CAIXA/day-9 exclusion, reviewed July-24
  merging, auto-trim labeling/decoding, and day parsing into the exporter.
- Removed AST loading and the runtime dependency on
  `train_dinov2_endpoints_kaggle.py`; the exporter can now be pasted into one
  Kaggle cell by itself.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-28] dataset | export both classification tasks
- The standalone materializer now reads the all-frame dataset root and exports
  both `obstacle_freepath` and `deviation_lrn`.
- Each task has independent `train_val` and `test` class folders; deviation
  therefore includes `left`, `right`, and `none` without mixing those labels
  into the obstacle/free-path class space.
- The combined root manifest adds a `task` column, and the summary reports
  counts independently for both tasks.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-28] dataset | store each frame once with two labels
- Replaced the duplicated per-task image trees with
  `train_val/images` and `test/images`; each physical frame is copied once.
- Root `labels.csv` now has `obstacle_label` and `deviation_label` columns for
  every image, plus split and source-clip provenance.
- Base and July-24 manifests are paired one-to-one by clip/frame identity.
  Auto-trim video is decoded once and both labels are derived from event
  metadata.
- Summary counts now describe unique images and both label distributions.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-28] dataset | add universal DINOv2 cosine selector
- Added a standalone Kaggle script that embeds the first 1,000 valid
  multi-label frames with frozen DINOv2 ViT-S/14.
- Greedy selection accepts a frame only when its cosine similarity is below
  0.90 for every previously accepted frame, guaranteeing the same pairwise
  constraint in the final dataset.
- Selected images preserve both labels and source clips. A full decision audit
  and verified maximum pairwise similarity accompany the output manifest.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-28] training | train both classifiers from selected images
- Added a Kaggle trainer that reads only the universal cosine selector output;
  no unselected source frame is discovered or loaded.
- Selected test membership remains unchanged. Selected train_val clips are
  split into train/validation with the established class-stratified clip policy.
- Frozen DINOv2 embeddings feed separate single-linear-layer obstacle and
  deviation classifiers. Existing deviation-only flip augmentation is retained.
- Outputs include both checkpoints and combined metrics with split provenance.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-28] training | print selected split class counts
- The selected-image trainer now prints per-class counts for source train,
  effective post-augmentation train, validation, and test for both classifiers.
- Effective deviation counts expose the label-swapped flip copies separately
  from the original selected training rows; the same counts are saved in
  metrics.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-28] dataset | bypass DINOv2 selection for test
- Universal cosine embedding and filtering now apply only to train_val rows.
- Every original test row is copied unchanged and is never opened by the
  DINOv2 embedding loader; the audit covers train_val candidates only.
- Summary output separately records selected/rejected train_val frames, fully
  preserved test frames, total output, and `dinov2_applied_to_test=false`.
- The selected-image trainer consequently evaluates on the complete preserved
  selected-dataset test split.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-28] training | enforce no class balancing
- The selected-image trainer explicitly records that balancing and class
  weights are disabled for both classifiers.
- Runtime assertions require the clip-grouped train/validation split to retain
  every selected source row exactly once; test membership remains untouched.
- Checkpoints, metrics, and console output expose the no-balancing invariant.
  Existing flip augmentation adds copies without sampling source rows.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-28] training | balance selected training backgrounds by clip rate
- Superseding the preceding no-balance setting, training now samples only
  background classes at evenly spaced per-clip intervals.
- Free-path targets the obstacle count; none targets the rounded mean of
  left/right. Every obstacle, left, and right source row is preserved.
- Validation and test remain unbalanced and untouched. No class weights or
  weighted sampler are introduced.
- Metrics/checkpoints record exact before/after totals and per-clip keep
  fractions/frame strides; console output prints every stage's class counts.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-28] training | apply the training balance policy to validation
- Validation now independently samples free-path and none frames with the same
  per-clip, evenly spaced policy used for training.
- Validation free-path matches its obstacle count; validation none matches the
  rounded mean of its left/right counts. Foreground rows remain intact.
- Validation is not augmented, and test remains naturally distributed and
  untouched.
- Metrics/checkpoints and console output expose validation counts before and
  after balancing plus its per-clip sampling details.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-28] training | add selected-image ImageNet MobileNetV1 trainer
- Added a standalone Kaggle trainer for the two-label similarity-selected
  dataset using two independently fine-tuned ImageNet MobileNetV1 models.
- It mirrors the DINOv2 trainer's authoritative test split, per-task
  clip-grouped train/validation split, independent background balancing, and
  training-only deviation flip policy.
- It prints every class count, saves best-validation checkpoints for both
  tasks, and writes combined histories, split metadata, and metrics.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-28] training | add merged all-frame MobileNetV1 trainer
- Added a standalone ImageNet MobileNetV1 trainer that reproduces the supplied
  DINOv2 script's runtime dataset construction and split.
- Base, July-24, and auto-trim frames follow the same CAIXA exclusion,
  duplicate avoidance, days 22/24 test, balancing, and flip rules.
- MobileNetV1 is fine-tuned end-to-end, restores its best validation-loss
  state, and saves the full build/split history beside `model.pt`.
- Pages touched: [[kaggle-image-datasets]].

## [2026-07-28] documentation | create Markdown paper working copy
- Created `WiP_2026_PT/artigo.md` from the Portuguese LaTeX draft without
  modifying `artigo.tex`.
- Corrected the model formulation to one binary obstacle head and one
  left/right/none head; forward and stop are decoded from the joint outputs.
- Documented all six valid output pairs. Clear+left/right remains actionable
  for corridor centering; the obstacle head disambiguates only none into
  forward or stop.
- Pages touched: [[kaggle-two-head-dataset]].

## [2026-07-28] documentation | detail automatic IMU selection in paper
- Replaced the inertial-method placeholder in `WiP_2026_PT/artigo.md` with the
  implemented gravity alignment, per-clip yaw/lateral fallback, persistence,
  stop/deceleration, quality, confidence, and temporal-window rules.
- Added primary references for gravity-based heading, gait autocorrelation,
  zero-velocity detection, and MEMS inertial drift.
- Explicitly distinguished SenseCV's empirical thresholds and simple
  rolling-energy/drift-removal implementation from the cited methods.
- Pages touched: [[imu-event-labeling]].

## [2026-07-28] documentation | generalize scientific IMU description
- Recast the inertial-selection section of `WiP_2026_PT/artigo.md` as a
  scientific methods description without calibration values or validation
  results.
- Consolidated the four inertial references into one broad methodological
  citation instead of assigning an individual source to each processing step.
- Pages touched: [[imu-event-labeling]].

## [2026-07-28] documentation | consolidate suggested-cut reference
- Added `WiP_2026_PT/referencia_cortes_sugeridos_imu.md` with the complete
  event-to-window-to-export path, relevant code excerpts, API and script
  consumers, wiki knowledge, internal methodology, and academic citations.
- Clarified the distinction between suggesting a temporal window, writing the
  MP4, and selecting visually distinct frames inside the cut.
- Corrected the free-walk documentation: detectable cadence qualifies the clip,
  while the current `_livre_cut()` returns the full `[0, duration]` interval.
- Reconciled later code changes: free-walk now requires any detectable cadence
  window, while the stop fallback locates cadence loss near the clip end;
  `IMU_STOP_ONSET_MIN_SEC` and `IMU_STOP_CONFIRM_SEC` are currently declared
  but unused.
- Pages touched: [[crop-suggestion]], [[imu-event-labeling]].
