# -*- coding: utf-8 -*-
"""Run the 30-FPS five-seed random-frame/random-clip experiment with MobileNetV4.

This reuses the verified MobileNetV1 random-split pipeline, replacing only the
model passed to the shared two-head trainer.  Outputs are written below
``/kaggle/working/mobilenetv4_two_head_random_splits``.
"""

import runpy
from pathlib import Path


KAGGLE_BASE = Path(
    "/kaggle/input/datasets/anamariabarrosnonato/scripthelper/"
    "mobilenetv1_two_head_random_splits_kaggle.py"
)
SCRIPT_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()


def base_path():
    local = SCRIPT_DIR / "mobilenetv1_two_head_random_splits_kaggle.py"
    if local.is_file():
        return local
    if KAGGLE_BASE.is_file():
        return KAGGLE_BASE
    raise FileNotFoundError("Upload mobilenetv1_two_head_random_splits_kaggle.py beside this file.")


def run(config_updates=None):
    experiment = runpy.run_path(str(base_path()), run_name="_mobilenetv4_random_splits")
    settings = {
        "output_dir": "/kaggle/working/mobilenetv4_two_head_random_splits",
        "target_fps": None,
    }
    if config_updates:
        settings.update(config_updates)
    experiment["CONFIG"].update(settings)
    # ``train_one`` is defined by the all-frames base trainer, which the
    # random-splits script loads inside its main().  Patch that one load rather
    # than looking for train_one in the random-splits module itself.
    original_run_path = experiment["runpy"].run_path

    def load_base_trainer(*args, **kwargs):
        base = original_run_path(*args, **kwargs)
        original_train_one = base["train_one"]

        def train_mobilenetv4(_unused_name, *train_args, **train_kwargs):
            return original_train_one("mobilenetv4", *train_args, **train_kwargs)

        base["train_one"] = train_mobilenetv4
        return base

    experiment["runpy"].run_path = load_base_trainer
    try:
        experiment["main"]()
    finally:
        experiment["runpy"].run_path = original_run_path


def main():
    run()


if __name__ == "__main__":
    main()
