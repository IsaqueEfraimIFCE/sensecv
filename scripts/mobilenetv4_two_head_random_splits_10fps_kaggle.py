# -*- coding: utf-8 -*-
"""Run the 10-FPS five-seed random-split experiment with MobileNetV4."""

import runpy
from pathlib import Path


KAGGLE_BASE = Path(
    "/kaggle/input/datasets/anamariabarrosnonato/scripthelper/"
    "mobilenetv4_two_head_random_splits_kaggle.py"
)
SCRIPT_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()


def main():
    local = SCRIPT_DIR / "mobilenetv4_two_head_random_splits_kaggle.py"
    base_path = local if local.is_file() else KAGGLE_BASE
    if not base_path.is_file():
        raise FileNotFoundError("Upload the MobileNetV4 30-FPS random-splits script beside this file.")
    experiment = runpy.run_path(str(base_path), run_name="_mobilenetv4_random_splits_10fps")
    experiment["run"]({
        "target_fps": 10.0,
        "source_fps": 30.0,
        "output_dir": "/kaggle/working/mobilenetv4_two_head_random_splits_10fps",
    })


if __name__ == "__main__":
    main()
