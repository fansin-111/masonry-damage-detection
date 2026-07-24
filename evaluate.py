"""Whole-patch evaluation for a trained segmentation model."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from utils.metrics import SegmentationMetrics
from utils.reproducibility import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate masonry point-cloud segmentation"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("weights/best_model.pth")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-points", type=int, default=4096)
    parser.add_argument("--num-votes", type=int, default=5)
    parser.add_argument("--block-size", type=float, default=0.5)
    parser.add_argument("--stride", type=float, default=0.5)
    parser.add_argument("--radius", type=float, default=0.005)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--save-predictions", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args()


def _load_state_dict(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def _add_votes(votes, point_indices, predictions, weights):
    valid = (weights != 0) & np.isfinite(weights)
    np.add.at(
        votes,
        (point_indices[valid].astype(np.int64), predictions[valid].astype(np.int64)),
        1,
    )


def main(args=None):
    args = parse_args() if args is None else args
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    seed_everything(args.seed, deterministic=True)
    device = torch.device(args.device)

    from datasets import DefectDatasetWholeScene
    from models import build_model

    dataset = DefectDatasetWholeScene(
        args.data_root,
        split="test",
        block_points=args.num_points,
        stride=args.stride,
        block_size=args.block_size,
    )
    model = build_model(args, num_classes=2).to(device)
    state_dict = _load_state_dict(args.checkpoint, device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_dir / f"evaluation-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    prediction_dir = run_dir / "predictions"
    if args.save_predictions:
        prediction_dir.mkdir()

    global_metrics = SegmentationMetrics(2)
    scene_rows = []
    with torch.no_grad():
        for scene_index, filename in enumerate(dataset.file_list):
            points = dataset.scene_points_list[scene_index]
            target = dataset.semantic_labels_list[scene_index]
            votes = np.zeros((target.shape[0], 2), dtype=np.int64)
            for _ in tqdm(
                range(args.num_votes), desc=filename, leave=False
            ):
                blocks, _, weights, point_indices = dataset[scene_index]
                for start in range(0, blocks.shape[0], args.batch_size):
                    end = min(start + args.batch_size, blocks.shape[0])
                    batch = torch.from_numpy(blocks[start:end]).float()
                    batch = batch.to(device).transpose(1, 2)
                    prediction = model(batch).argmax(dim=1).cpu().numpy()
                    _add_votes(
                        votes,
                        point_indices[start:end],
                        prediction,
                        weights[start:end],
                    )

            probabilities = votes / np.maximum(votes.sum(axis=1, keepdims=True), 1)
            prediction = (probabilities[:, 1] > args.threshold).astype(np.int64)
            scene_metrics = SegmentationMetrics(2)
            scene_metrics.update(prediction, target)
            result = scene_metrics.compute()
            global_metrics.update(prediction, target)
            scene_rows.append(
                {
                    "file": filename,
                    "accuracy": result["accuracy"],
                    "iou_0": result["class_iou"][0],
                    "iou_1": result["class_iou"][1],
                    "miou": result["miou"],
                }
            )
            if args.save_predictions:
                output = np.column_stack([points, prediction])
                np.save(prediction_dir / filename, output)

    result = global_metrics.compute()
    summary = {
        "checkpoint": str(args.checkpoint),
        "model": "improved_pointnetpp",
        "test_patches": len(dataset),
        "accuracy": result["accuracy"],
        "mean_class_accuracy": result["mean_class_accuracy"],
        "iou_0": float(result["class_iou"][0]),
        "iou_1": float(result["class_iou"][1]),
        "miou": result["miou"],
        "seed": args.seed,
        "num_votes": args.num_votes,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (run_dir / "per_patch.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=scene_rows[0].keys())
        writer.writeheader()
        writer.writerows(scene_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
