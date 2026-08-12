# -*- coding: utf-8 -*-
"""Evaluate final four-command route accuracy for MobileNetV4 at 30 and 10 FPS."""

import runpy
from pathlib import Path


EXPERIMENTS = [
    (
        "30fps",
        "/kaggle/input/datasets/anamariabarrosnonato/data234/"
        "mobilenetv4_two_head_random_splits",
    ),
    (
        "10fps",
        "/kaggle/input/datasets/anamariabarrosnonato/data234/"
        "mobilenetv4_two_head_random_splits_10fps",
    ),
]
KAGGLE_EVALUATOR = Path(
    "/kaggle/input/datasets/anamariabarrosnonato/scripthelper/"
    "evaluate_random_splits_route_accuracy.py"
)
OUTPUT_ROOT = Path("/kaggle/working")
SCRIPT_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()


def writable_evaluator_path(evaluator_path):
    source = evaluator_path.read_text(encoding="utf-8")
    output_statement = '    output_path = model_root / "final_route_accuracy.json"'
    if output_statement not in source:
        raise RuntimeError("The shared evaluator does not contain the expected report output statement.")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    runtime_evaluator = OUTPUT_ROOT / "evaluate_random_splits_route_accuracy_runtime.py"
    runtime_evaluator.write_text(
        source.replace(output_statement, '    output_path = Path(CONFIG["output_path"])'),
        encoding="utf-8",
    )
    return runtime_evaluator


def main():
    local_evaluator = SCRIPT_DIR / "evaluate_random_splits_route_accuracy.py"
    evaluator_path = local_evaluator if local_evaluator.is_file() else KAGGLE_EVALUATOR
    if not evaluator_path.is_file():
        raise FileNotFoundError(
            "Cannot find evaluate_random_splits_route_accuracy.py locally or in the Kaggle input dataset."
        )
    runtime_evaluator = writable_evaluator_path(evaluator_path)
    for label, model_root in EXPERIMENTS:
        output_path = OUTPUT_ROOT / Path(model_root).name / "final_route_accuracy.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        evaluator = runpy.run_path(str(runtime_evaluator), run_name=f"_mobilenetv4_route_{label}")
        evaluator["CONFIG"].update({
            "model_root": model_root,
            "model_dir_name": "mobilenetv4",
            "output_path": str(output_path),
        })
        print(f"\n=== MobileNetV4 {label} route evaluation ===")
        evaluator["main"]()


if __name__ == "__main__":
    main()
