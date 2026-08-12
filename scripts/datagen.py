# -*- coding: utf-8 -*-
"""Generate the exact Kaggle ``dataset_used`` export for the ConvLSTM run.

This is the data-only companion to ``convlstm_mobilenetv1_all_frames.py``.
It deliberately executes the trainer's own ingestion, clip split, sequence
construction, background balancing, flip augmentation, and export code, then
stops immediately before model setup. Therefore its ``dataset_used`` directory
is exactly the one a normal ConvLSTM run would write with the same CONFIG and
source attachments.

Kaggle usage:

    !python /kaggle/input/<your-code>/datagen.py

By default it uses the attached helper at
``/kaggle/input/datasets/anamariabarrosnonato/scripthelper/``. If that path is
not present (for example, when run locally), it falls back to a sibling
``convlstm_mobilenetv1_all_frames.py``. Edit CONFIG in the trainer first. In particular,
``output_dir`` controls where Kaggle writes ``dataset_used``. The default is
``/kaggle/working/convlstm_mobilenetv1_all_frames/dataset_used``.
"""

import os
import sys
from pathlib import Path


KAGGLE_TRAINER = Path(
    "/kaggle/input/datasets/anamariabarrosnonato/scripthelper/"
    "convlstm_mobilenetv1_all_frames.py"
)
SIBLING_TRAINER = (
    Path(__file__).with_name("convlstm_mobilenetv1_all_frames.py")
    if "__file__" in globals()
    else Path.cwd() / "convlstm_mobilenetv1_all_frames.py"
)


def data_only_source(trainer):
    """Return trainer code with a data-only exit if it predates this helper."""
    source = trainer.read_text(encoding="utf-8")
    if "SENSECV_CONVLSTM_DATA_ONLY" in source:
        return source

    # Kaggle input attachments are read-only. Insert the same exit immediately
    # after save_dataset_used() has run, without changing the attached script.
    marker = "\nfor split_name, split_samples in (\n"
    if source.count(marker) != 1:
        raise RuntimeError(
            "Unable to find the ConvLSTM training boundary in "
            f"{trainer}; use the matching convlstm_mobilenetv1_all_frames.py."
        )
    if "import os\n" not in source:
        source = source.replace("import json\n", "import json\nimport os\n", 1)
    guard = (
        "\nif os.environ.get(\"SENSECV_CONVLSTM_DATA_ONLY\") == \"1\":\n"
        "    print(\"done | generated exact ConvLSTM dataset; training skipped\")\n"
        "    raise SystemExit(0)\n"
    )
    return source.replace(marker, guard + marker, 1)


def main():
    trainer = KAGGLE_TRAINER if KAGGLE_TRAINER.is_file() else SIBLING_TRAINER
    if not trainer.is_file():
        raise FileNotFoundError(
            "ConvLSTM trainer not found. Expected either "
            f"{KAGGLE_TRAINER} or {SIBLING_TRAINER}."
        )
    previous = os.environ.get("SENSECV_CONVLSTM_DATA_ONLY")
    original_timm = sys.modules.get("timm")
    os.environ["SENSECV_CONVLSTM_DATA_ONLY"] = "1"
    # Older attached trainers import/install timm before the injected exit,
    # even though generation never reaches a model call. A placeholder keeps
    # that import from triggering a network installation.
    sys.modules.setdefault("timm", object())
    try:
        globals_dict = {"__name__": "__main__", "__file__": str(trainer)}
        exec(compile(data_only_source(trainer), str(trainer), "exec"), globals_dict)
    except SystemExit as exit_status:
        if exit_status.code not in (None, 0):
            raise
    finally:
        if original_timm is None:
            sys.modules.pop("timm", None)
        else:
            sys.modules["timm"] = original_timm
        if previous is None:
            os.environ.pop("SENSECV_CONVLSTM_DATA_ONLY", None)
        else:
            os.environ["SENSECV_CONVLSTM_DATA_ONLY"] = previous


if __name__ == "__main__":
    main()
