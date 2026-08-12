# -*- coding: utf-8 -*-
"""Run the five-seed random-frame/random-clip MobileNetV1 experiment at 10 FPS.

This companion uses the 60/20/20 experiment in
``mobilenetv1_two_head_random_splits_kaggle.py`` and samples the datagen image
export at 10 FPS. It assumes the source all-frame export is 30 FPS, so it keeps
every third chronological original frame. The result remains ten trainings:
five random-frame splits plus five whole-clip splits.
"""

import runpy
from pathlib import Path


KAGGLE_BASE = Path(
    "/kaggle/input/datasets/anamariabarrosnonato/scripthelper/"
    "mobilenetv1_two_head_random_splits_kaggle.py"
)
LOCAL_BASE = (
    Path(__file__).with_name("mobilenetv1_two_head_random_splits_kaggle.py")
    if "__file__" in globals()
    else Path.cwd() / "mobilenetv1_two_head_random_splits_kaggle.py"
)


def main():
    base_path = KAGGLE_BASE if KAGGLE_BASE.is_file() else LOCAL_BASE
    if not base_path.is_file():
        raise FileNotFoundError(
            "Random-splits base script not found. Upload "
            "mobilenetv1_two_head_random_splits_kaggle.py beside this file, "
            "or set KAGGLE_BASE to its attachment path."
        )
    base = runpy.run_path(str(base_path), run_name="_sensecv_random_splits")
    base["CONFIG"].update({
        "target_fps": 10.0,
        "source_fps": 30.0,
        "output_dir": "/kaggle/working/mobilenetv1_two_head_random_splits_10fps",
    })
    base["main"]()


if __name__ == "__main__":
    main()
