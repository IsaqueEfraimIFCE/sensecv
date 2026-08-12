# -*- coding: utf-8 -*-
"""Evaluate saved random-split MobileNetV1 checkpoints as route commands.

Route definition:
  free_path + none -> forward
  obstacle  + none -> stop
  either obstacle state + right -> right
  either obstacle state + left  -> left

Therefore an obstacle may still have a left/right deviation, and a clear-path
frame may use left/right to re-centre in the corridor.

Set MODEL_ROOT to the folder containing summary.json and checkpoint folders:
<MODEL_ROOT>/<random_frames|random_clips>_seed_<seed>/mobilenetv1/model.pt
"""

import json
import runpy
from pathlib import Path


CONFIG = {
    "model_root": "/kaggle/working/mobilenetv1_two_head_random_splits",
    "dataset_used_dir": None,  # None uses the location recorded in summary.json.
    "random_splits_script": None,
    "base_trainer": None,
    "model_dir_name": "mobilenetv1",  # Checkpoint subdirectory within each seed run.
    "batch_size": 64,
    "num_workers": 2,
}

KAGGLE_RANDOM_SPLITS = Path(
    "/kaggle/input/datasets/anamariabarrosnonato/scripthelper/"
    "mobilenetv1_two_head_random_splits_kaggle.py"
)
ROUTE_CLASSES = ["forward", "right", "left", "stop"]
SCRIPT_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()


def find_script(configured, filename, kaggle_fallback=None):
    path = Path(configured) if configured else SCRIPT_DIR / filename
    if path.is_file():
        return path
    if kaggle_fallback and kaggle_fallback.is_file():
        return kaggle_fallback
    raise FileNotFoundError(f"Cannot find {filename}; set its path in CONFIG.")


def route_indices(obstacle, deviation, obstacle_classes, deviation_classes):
    """Convert batches of two-head class indices to forward/right/left/stop."""
    torch = __import__("torch")
    route = torch.empty_like(obstacle)
    left = deviation_classes.index("left")
    right = deviation_classes.index("right")
    none = deviation_classes.index("none")
    obstacle_present = obstacle == obstacle_classes.index("obstacle")
    route[deviation == left] = ROUTE_CLASSES.index("left")
    route[deviation == right] = ROUTE_CLASSES.index("right")
    route[(deviation == none) & ~obstacle_present] = ROUTE_CLASSES.index("forward")
    route[(deviation == none) & obstacle_present] = ROUTE_CLASSES.index("stop")
    return route


def evaluate_route(model, test_rows, transform, base, device, seed):
    torch = base["torch"]
    data_loader = base["loader"](test_rows, transform, device, False, False, seed)
    obstacle_classes = base["OBSTACLE_CLASSES"]
    deviation_classes = base["DEVIATION_CLASSES"]
    matrix = [[0] * len(ROUTE_CLASSES) for _ in ROUTE_CLASSES]
    model.eval()
    with torch.inference_mode():
        for images, obstacle, deviation, _, _ in data_loader:
            images, obstacle, deviation = images.to(device), obstacle.to(device), deviation.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                obstacle_logits, deviation_logits = model(images)
            truth = route_indices(obstacle, deviation, obstacle_classes, deviation_classes)
            predicted = route_indices(obstacle_logits.argmax(1), deviation_logits.argmax(1), obstacle_classes, deviation_classes)
            for actual, estimate in zip(truth.cpu().tolist(), predicted.cpu().tolist()):
                matrix[actual][estimate] += 1
    total = sum(map(sum, matrix))
    correct = sum(matrix[index][index] for index in range(len(ROUTE_CLASSES)))
    return {
        "frames": total,
        "route_accuracy": round(correct / max(1, total), 6),
        "route_classes": ROUTE_CLASSES,
        "confusion_matrix": matrix,  # rows=true command, columns=predicted command
        "true_route_counts": dict(zip(ROUTE_CLASSES, map(sum, matrix))),
    }


def main():
    model_root = Path(CONFIG["model_root"])
    summary_path = model_root / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Expected summary.json at {summary_path}")
    experiment = json.loads(summary_path.read_text(encoding="utf-8"))

    split_path = find_script(CONFIG["random_splits_script"], "mobilenetv1_two_head_random_splits_kaggle.py", KAGGLE_RANDOM_SPLITS)
    splits = runpy.run_path(str(split_path), run_name="_route_split_source")
    trainer_path = Path(CONFIG["base_trainer"]) if CONFIG["base_trainer"] else splits["base_trainer_path"]()
    base = runpy.run_path(str(trainer_path), run_name="_route_base_trainer")

    sampling = experiment.get("frame_sampling", {})
    config = experiment.get("config", {})
    rows = splits["sample_to_fps"](
        splits["load_all_frames"](CONFIG["dataset_used_dir"] or experiment["dataset_used_dir"]),
        sampling.get("target_fps", config.get("target_fps")),
        sampling.get("source_fps_assumption", config.get("source_fps", 30.0)),
    )
    image_size = config.get("image_size", 224)
    base["CONFIG"].update({
        "image_size": image_size,
        "batch_size": CONFIG["batch_size"],
        "num_workers": CONFIG["num_workers"],
        "pretrained": False,
    })
    transform = base["transforms"].Compose([
        base["transforms"].Resize((image_size, image_size), base["transforms"].InterpolationMode.BICUBIC),
        base["transforms"].ToTensor(),
        base["transforms"].Normalize(base["MEAN"], base["STD"]),
    ])
    device = base["torch"].device("cuda" if base["torch"].cuda.is_available() else "cpu")
    report = {
        "route_definition": {
            "forward": "free_path + none",
            "right": "right, with or without obstacle",
            "left": "left, with or without obstacle",
            "stop": "obstacle + none",
        },
        "runs": {},
    }

    splitter_names = {"random_frames": "random_frame_split", "random_clips": "random_clip_split"}
    for run_name in sorted(experiment["runs"]):
        mode, _, seed_text = run_name.rpartition("_seed_")
        seed = int(seed_text)
        checkpoint_path = model_root / run_name / CONFIG["model_dir_name"] / "model.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
        _, _, test_rows = splits[splitter_names[mode]](rows, seed)
        checkpoint = base["torch"].load(checkpoint_path, map_location=device, weights_only=False)
        model = base["make_model"](checkpoint["architecture"], device)
        model.load_state_dict(checkpoint["state_dict"])
        metrics = evaluate_route(model, test_rows, transform, base, device, seed)
        report["runs"][run_name] = metrics
        print(f"{run_name}: route accuracy={metrics['route_accuracy']:.4%} ({metrics['frames']} frames)")

    for mode in splitter_names:
        scores = [entry["route_accuracy"] for name, entry in report["runs"].items() if name.startswith(mode + "_seed_")]
        if scores:
            mean = sum(scores) / len(scores)
            std = (sum((score - mean) ** 2 for score in scores) / len(scores)) ** 0.5
            report.setdefault("summary", {})[mode] = {
                "runs": len(scores), "mean_route_accuracy": round(mean, 6), "population_std": round(std, 6)
            }

    output_path = model_root / "final_route_accuracy.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
