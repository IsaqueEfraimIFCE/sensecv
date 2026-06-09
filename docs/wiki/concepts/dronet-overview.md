# DroNet Overview

This workspace adapts DroNet, a 2018 neural navigation system, to run offline inference on exported PilotGuru supermarket videos.

## What Is Here

- `dronet/repo/` contains the upstream `rpg_public_dronet` code, including Keras training/evaluation code, pretrained model weights, model JSON, and ROS packages for Bebop drone perception/control.
- `dronet/dronet_model.py` is a local PyTorch reimplementation of the original Keras ResNet-8 inference graph.
- `dronet/run_dronet.py` runs the PyTorch model frame by frame over videos from `exports/` by default.
- `dronet_results/` contains per-clip CSVs, annotated videos, a combined frame table, example crops, and aggregate `summary.json`.
- `dronet/paper_text.txt` and `dronet/RAL18_Loquercio.pdf` are the paper source layer.

## Current Synthesis

The local implementation is designed to preserve the original DroNet inference behavior while avoiding the old TensorFlow/Keras runtime constraint. The key compatibility details are TensorFlow-style `SAME` padding, Keras channels-last flattening, BatchNorm epsilon `1e-3`, and the same grayscale resize/crop/rescale preprocessing.

The supermarket experiment is intentionally out of distribution. DroNet was trained on forward-facing car and bicycle imagery from streets and collision sequences, while the local clips are 4K portrait phone videos inside a supermarket. The high collision probabilities in [[experiment-results]] should be treated as raw model responses, not calibrated safety judgments.

## Primary Caveat

The local code produces useful exploratory outputs, but the model has not been validated for this indoor supermarket domain. Any operational use would require calibration, domain-specific evaluation data, and safety testing.
