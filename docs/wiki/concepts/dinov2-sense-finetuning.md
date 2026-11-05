---
type: concept
tags: [training, dinov2, experiment, fine-tuning, kaggle]
code_refs: [scripts/train_dinov2_sense_fps5_uniao_kaggle.py]
updated: 2026-08-17
---

# DINOv2 fine-tuning ladder (5 FPS uniao)

Three DINOv2 ViT-S/14 two-head runs on **exactly** the fps5/uniao data path of
[[mobilenetv4-sense-experiments]] — 8,029 training frames, 1,378 validation,
5 FPS on every split, union of distinct + extended-window frames. The backbone
and what is unfrozen are the only differences from the MobileNetV4 run, so the
comparison is clean.

Configuration is run4's throughout: 224x224, `Resize(BICUBIC)` + `ToTensor` +
ImageNet `Normalize`, batch 32, AdamW `lr 1e-4` / `wd 1e-4`,
`ReduceLROnPlateau(min, 0.5, patience 2)`, fp16 AMP, 40 epochs, patience 6,
best state by validation loss, seed 43, hflip on train only,
`loss = CE(obstacle) + CE(deviation)` on every frame.

## Results

Joint accuracy. T1 = original distinct test (3,970), T2 = its 5 FPS subsample
(704), T4 = T1 + all janela test frames (6,676).

| run | trainable | T2 | T1 | janela 5 FPS | T4 | best ep | min |
|---|---|---|---|---|---|---|---|
| **ultimos1** | 1,777,925 (8.1%) | 0.8864 | **0.8962** | **0.9522** | **0.9194** | 6/12 | 11.2 |
| ultimos2 | 3,553,157 (16.1%) | **0.8892** | 0.8854 | 0.8773 | 0.8833 | 5/11 | 10.9 |
| completo | 22,058,501 (100%) | 0.4517 | 0.4426 | 0.0104 | 0.2681 | 7/13 | 27.4 |

Per-class F1 on T1:

| run | free_path | obstacle | left | right | none |
|---|---|---|---|---|---|
| ultimos1 | 0.880 | 0.923 | 0.929 | 0.900 | 0.880 |
| ultimos2 | 0.947 | 0.959 | 0.874 | 0.894 | 0.892 |
| completo | 0.621 | 0.098 | 0.000 | 0.037 | 0.615 |

## Findings

- **Unfreezing one block is enough, and adding a second does not help.**
  ultimos1 beats ultimos2 on T1 (0.8962 vs 0.8854) and clearly on T4 (0.9194 vs
  0.8833) with half the trainable parameters. The frozen DINOv2 representation
  is already close to sufficient; the final block only needs re-aiming.
- **DINOv2 ultimos1 is the best model on T4 of every model trained so far**
  (0.9194 against 0.8607 for the best MobileNetV4, full_rate/uniao) — while
  training on 8,029 frames instead of 34,394.
- **It rescues the fps5/uniao collapse.** MobileNetV4 on this identical data
  reached joint 0.5861 on T1 with best epoch 2; DINOv2 ultimos1 reaches 0.8962.
  The collapse was an optimization failure, not a data limitation: a stronger
  frozen representation converges before early stopping intervenes.
- **The full fine-tune did not diverge — it never finished converging.**
  Train loss fell monotonically 1.144 -> 0.180 and validation deviation macro-F1
  was *still rising* at the last epoch (0.502 -> 0.885) when patience ran out at
  epoch 13. The degenerate test predictions (left F1 0.000, obstacle F1 0.098)
  are what an undertrained ViT looks like, not a diverged one.
  **`lr 1e-4` is roughly 10x too high for a full ViT-S fine-tune with AdamW.**
  The available conclusion is "full fine-tuning does not converge at run4's LR
  within 40 epochs / patience 6", not "full fine-tuning is worse". A rerun at
  `lr 1e-5`, or with warmup, would settle it. **Not yet run.**

## Error structure differs from MobileNetV4

The MobileNetV4 models essentially never invert direction; their errors are a
fallback to `none`. DINOv2 ultimos2 makes genuine inversions (77 left->right on
T1). ultimos1 is better behaved: 11 left->right and 4 right->left, with the
residual error being `none` misread as a direction (174 none->left,
185 none->right).

For assistive navigation the distinction matters — a `none` fallback is silent,
an inversion steers the person the wrong way. Worth stating in the paper
alongside accuracy.

## Reproducing

`scripts/train_dinov2_sense_fps5_uniao_kaggle.py`. The strategy name is parsed
as `ultimosN`, so any depth is a one-line `CONFIG["runs"]` change. Kernels:
`sensecv-dinov2-5-fps-uniao` (ultimos2 + completo) and
`sensecv-dinov2-5-fps-uniao-last-1`.

## Related

- [[mobilenetv4-sense-experiments]]
- [[sense-style-datasets]]
- [[kaggle-image-datasets]] — the project's earlier DINOv2 trainers
