"""Train the final improved PointNet++ model."""

import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.augment import jitter, random_scale, random_shift, rotate_z
from utils.metrics import SegmentationMetrics
from utils.reproducibility import seed_everything, seed_worker


NUM_CLASSES = 2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train masonry damage point-wise segmentation"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-points", type=int, default=4096)
    parser.add_argument("--block-size", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--lr-decay", type=float, default=0.7)
    parser.add_argument("--step-size", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument(
        "--radius",
        type=float,
        default=0.005,
        help="Base squared-radius parameter used by NSA grouping",
    )
    parser.add_argument("--focal-gamma", type=float, default=3.0)
    parser.add_argument("--label-smoothing", type=float, default=0.2)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--drop-last-val",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match the released experiment, which dropped the last partial val batch",
    )
    parser.add_argument(
        "--rotate", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--jitter", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--scale", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--shift", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def focal_smoothed_loss(
    prediction,
    target,
    sample_weights=None,
    smoothing=0.2,
    focal_gamma=3.0,
):
    target = target.contiguous().view(-1)
    class_count = prediction.shape[1]
    one_hot = torch.zeros_like(prediction).scatter_(
        1, target.unsqueeze(1), 1
    )
    if smoothing:
        one_hot = one_hot * (1 - smoothing) + (1 - one_hot) * (
            smoothing / (class_count - 1)
        )
    log_probability = F.log_softmax(prediction, dim=1)
    loss = -(one_hot * log_probability).sum(dim=1)
    if focal_gamma > 0:
        true_probability = torch.exp(
            log_probability.gather(1, target.unsqueeze(1))
        ).squeeze(1)
        loss = loss * (1 - true_probability).pow(focal_gamma)
    if sample_weights is not None:
        loss = loss * sample_weights.reshape(-1).float()
    return loss.mean()


def _configure_logger(run_dir):
    logger = logging.getLogger("masonry_damage")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def _set_bn_momentum(module, momentum):
    if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
        module.momentum = momentum


def _augment(points, args):
    points = points.numpy()
    if args.rotate:
        points[:, :, :3] = rotate_z(points[:, :, :3])
    if args.jitter:
        points[:, :, :3] = jitter(points[:, :, :3])
    if args.scale:
        points[:, :, :3] = random_scale(points[:, :, :3])
    if args.shift:
        points[:, :, :3] = random_shift(points[:, :, :3])
    return torch.from_numpy(points).float()


def _run_epoch(model, loader, device, args, optimizer=None):
    training = optimizer is not None
    model.train(training)
    metrics = SegmentationMetrics(NUM_CLASSES)
    total_loss = 0.0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for points, target, sample_weights in tqdm(loader, leave=False):
            if training:
                points = _augment(points, args)
                optimizer.zero_grad(set_to_none=True)
            points = points.to(device, non_blocking=True).transpose(1, 2)
            target = target.long().to(device, non_blocking=True)
            sample_weights = sample_weights.to(device, non_blocking=True)
            logits = model(points).permute(0, 2, 1).contiguous()
            flat_logits = logits.view(-1, NUM_CLASSES)
            loss = focal_smoothed_loss(
                flat_logits,
                target,
                sample_weights if training else None,
                smoothing=args.label_smoothing,
                focal_gamma=args.focal_gamma if training else 0.0,
            )
            if training:
                loss.backward()
                optimizer.step()
            prediction = logits.argmax(dim=-1)
            metrics.update(prediction.detach().cpu().numpy(), target.cpu().numpy())
            total_loss += loss.item()
    result = metrics.compute()
    result["loss"] = total_loss / max(len(loader), 1)
    return result


def main(args=None):
    args = parse_args() if args is None else args
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    seed_everything(args.seed, deterministic=args.deterministic)
    device = torch.device(args.device)

    from datasets import DefectDataset
    from models import build_model

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"improved-pointnetpp-{timestamp}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    logger = _configure_logger(run_dir)
    (run_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8"
    )

    train_dataset = DefectDataset(
        "train", args.data_root, args.num_points, args.block_size
    )
    val_dataset = DefectDataset(
        "val", args.data_root, args.num_points, args.block_size
    )
    generator = torch.Generator().manual_seed(args.seed)
    common_loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=True, **common_loader_args
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=args.drop_last_val,
        **common_loader_args,
    )

    model = build_model(args, NUM_CLASSES).to(device)
    model.apply(
        lambda module: setattr(module, "inplace", True)
        if isinstance(module, torch.nn.ReLU)
        else None
    )
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("model=improved_pointnetpp trainable_parameters=%d", parameter_count)
    logger.info(
        "train_samples=%d val_samples=%d", len(train_dataset), len(val_dataset)
    )
    writer = SummaryWriter(run_dir / "tensorboard")
    csv_path = run_dir / "metrics.csv"
    best_miou = -np.inf
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "epoch",
            "learning_rate",
            "train_loss",
            "train_accuracy",
            "train_miou",
            "val_loss",
            "val_accuracy",
            "val_iou_0",
            "val_iou_1",
            "val_miou",
            "best_val_miou",
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        for epoch in range(args.epochs):
            learning_rate = max(
                args.learning_rate * args.lr_decay ** (epoch // args.step_size),
                1e-5,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            momentum = max(0.1 * 0.5 ** (epoch // args.step_size), 0.01)
            model.apply(lambda module: _set_bn_momentum(module, momentum))

            train_result = _run_epoch(
                model, train_loader, device, args, optimizer=optimizer
            )
            val_result = _run_epoch(model, val_loader, device, args)
            is_best = val_result["miou"] >= best_miou
            best_miou = max(best_miou, val_result["miou"])
            state_dict = model.state_dict()
            torch.save(state_dict, run_dir / "latest_model.pth")
            if is_best:
                torch.save(state_dict, run_dir / "best_model.pth")

            row = {
                "epoch": epoch + 1,
                "learning_rate": learning_rate,
                "train_loss": train_result["loss"],
                "train_accuracy": train_result["accuracy"],
                "train_miou": train_result["miou"],
                "val_loss": val_result["loss"],
                "val_accuracy": val_result["accuracy"],
                "val_iou_0": val_result["class_iou"][0],
                "val_iou_1": val_result["class_iou"][1],
                "val_miou": val_result["miou"],
                "best_val_miou": best_miou,
            }
            csv_writer.writerow(row)
            csv_file.flush()
            for key, value in row.items():
                if key != "epoch":
                    writer.add_scalar(key, value, epoch + 1)
            logger.info(
                "epoch=%03d train_acc=%.4f val_acc=%.4f val_mIoU=%.4f best=%.4f",
                epoch + 1,
                train_result["accuracy"],
                val_result["accuracy"],
                val_result["miou"],
                best_miou,
            )
    writer.close()
    logger.info("Training complete: %s", run_dir)


if __name__ == "__main__":
    main()
