"""Train and test M2Heat on Trento, Houston2013, and Augsburg.

Examples:
    python demo.py --dataset trento --device cuda --gpu-id 0
    python demo.py --dataset all --device cuda --gpu-id 0
    python demo.py --dataset trento --mode test --checkpoint log/<run>/best.pt
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import platform
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix

from m2heat import M2Heat


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    directory: str
    hsi_file: str
    hsi_key: str
    auxiliary_file: str
    auxiliary_key: str
    train_label_file: str
    train_label_key: str
    test_label_file: str
    test_label_key: str
    learning_rate: float
    batch_size: int
    patch_size: int


# These defaults are the settings used by the original three M2Heat demos.
DATASETS = {
    "trento": DatasetConfig(
        "Trento", "Trento", "trento_hsi.mat", "HSI", "trento_lidar.mat", "LiDAR",
        "TrainImage.mat", "train_gt", "TestImage.mat", "test_gt", 5e-4, 64, 13,
    ),
    "houston2013": DatasetConfig(
        "Houston2013", "Houston2013", "data_HS_HR.mat", "data_HS_HR", "data_DSM_HR.mat", "DSM",
        "TrainImage.mat", "TrainImage", "TestImage.mat", "TestImage", 5e-4, 64, 13,
    ),
    "augsburg": DatasetConfig(
        "Augsburg", "Augsburg", "data_HS_LR.mat", "data_HS_LR", "data_DSM.mat", "data_DSM",
        "TrainImage.mat", "TrainImage", "TestImage.mat", "TestImage", 5e-4, 64, 13,
    ),
}


@dataclass
class DatasetBundle:
    config: DatasetConfig
    input_train: torch.Tensor
    target_train: torch.Tensor
    input_test: torch.Tensor
    target_test: torch.Tensor
    hsi_bands: int
    auxiliary_bands: int
    num_classes: int
    height: int
    width: int


class Tee:
    def __init__(self, stream, file_handle):
        self.stream = stream
        self.file_handle = file_handle

    def write(self, text: str) -> None:
        self.stream.write(text)
        self.file_handle.write(text)

    def flush(self) -> None:
        self.stream.flush()
        self.file_handle.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M2Heat training and testing demo")
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint used by --mode test")
    parser.add_argument("--data-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--output-root", type=Path, default=ROOT / "log")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpu-id", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def select_device(kind: str, gpu_id: int) -> torch.device:
    if kind == "cpu" or (kind == "auto" and not torch.cuda.is_available()):
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
        raise ValueError(f"GPU id {gpu_id} is unavailable; found {torch.cuda.device_count()} GPU(s)")
    torch.cuda.set_device(gpu_id)
    return torch.device("cuda", gpu_id)


def normalize_hsi(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    minimum = image.min(axis=(0, 1), keepdims=True)
    maximum = image.max(axis=(0, 1), keepdims=True)
    denominator = np.where(maximum > minimum, maximum - minimum, 1.0)
    return (image - minimum) / denominator


def mirror_image(image: np.ndarray, patch_size: int) -> np.ndarray:
    padding = patch_size // 2
    height, width, bands = image.shape
    mirrored = np.empty((height + 2 * padding, width + 2 * padding, bands), dtype=np.float32)
    mirrored[padding:padding + height, padding:padding + width] = image
    for index in range(padding):
        mirrored[padding:padding + height, index] = image[:, padding - index - 1]
        mirrored[padding:padding + height, width + padding + index] = image[:, width - index - 1]
    for index in range(padding):
        mirrored[index] = mirrored[2 * padding - index - 1]
        mirrored[height + padding + index] = mirrored[height + padding - index - 1]
    return mirrored


def labeled_points(label: np.ndarray) -> np.ndarray:
    points = [np.argwhere(label == class_id) for class_id in range(1, int(label.max()) + 1)]
    points = [item for item in points if item.size]
    if not points:
        raise ValueError("The label image does not contain any foreground class")
    return np.concatenate(points, axis=0).astype(np.int64)


def extract_patches(
    mirrored: np.ndarray,
    label: np.ndarray,
    points: np.ndarray,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    patches = np.empty((len(points), patch_size, patch_size, mirrored.shape[2]), dtype=np.float32)
    targets = np.empty(len(points), dtype=np.int64)
    for index, (row, column) in enumerate(points):
        patches[index] = mirrored[row:row + patch_size, column:column + patch_size]
        targets[index] = int(label[row, column]) - 1
    return patches, targets


def load_dataset(name: str, data_root: Path, patch_size: int) -> DatasetBundle:
    config = DATASETS[name]
    data_dir = data_root / config.directory
    hsi = loadmat(data_dir / config.hsi_file)[config.hsi_key]
    auxiliary = loadmat(data_dir / config.auxiliary_file)[config.auxiliary_key]
    train_label = loadmat(data_dir / config.train_label_file)[config.train_label_key]
    test_label = loadmat(data_dir / config.test_label_file)[config.test_label_key]

    hsi = normalize_hsi(hsi)
    auxiliary = np.asarray(auxiliary, dtype=np.float32)
    if auxiliary.ndim == 2:
        auxiliary = auxiliary[..., None]
    train_label = np.asarray(train_label).squeeze()
    test_label = np.asarray(test_label).squeeze()
    if hsi.shape[:2] != auxiliary.shape[:2] or hsi.shape[:2] != train_label.shape:
        raise ValueError(
            f"Shape mismatch for {config.name}: HSI={hsi.shape}, auxiliary={auxiliary.shape}, "
            f"train labels={train_label.shape}"
        )

    combined = np.concatenate((hsi, auxiliary), axis=2)
    mirrored = mirror_image(combined, patch_size)
    train_points = labeled_points(train_label)
    test_points = labeled_points(test_label)
    train_patches, train_targets = extract_patches(mirrored, train_label, train_points, patch_size)
    test_patches, test_targets = extract_patches(mirrored, test_label, test_points, patch_size)
    num_classes = int(max(train_label.max(), test_label.max()))

    return DatasetBundle(
        config=config,
        input_train=torch.from_numpy(train_patches.transpose(0, 3, 1, 2)),
        target_train=torch.from_numpy(train_targets),
        input_test=torch.from_numpy(test_patches.transpose(0, 3, 1, 2)),
        target_test=torch.from_numpy(test_targets),
        hsi_bands=hsi.shape[2],
        auxiliary_bands=auxiliary.shape[2],
        num_classes=num_classes,
        height=hsi.shape[0],
        width=hsi.shape[1],
    )


def make_loader(
    bundle: DatasetBundle,
    split: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> data.DataLoader:
    if split == "train":
        dataset = data.TensorDataset(bundle.input_train, bundle.target_train)
        return data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    dataset = data.TensorDataset(bundle.input_test, bundle.target_test)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)


def metrics(targets: Sequence[int], predictions: Sequence[int], num_classes: int) -> dict[str, object]:
    matrix = confusion_matrix(targets, predictions, labels=list(range(num_classes)))
    total = int(matrix.sum())
    class_accuracy = np.divide(
        np.diag(matrix), matrix.sum(axis=1), out=np.zeros(num_classes, dtype=np.float64), where=matrix.sum(axis=1) != 0
    )
    oa = float(np.trace(matrix) / total) if total else 0.0
    expected = float((matrix.sum(axis=0) * matrix.sum(axis=1)).sum() / total**2) if total else 0.0
    kappa = float((oa - expected) / (1.0 - expected)) if abs(1.0 - expected) > 1e-12 else 0.0
    return {"OA": oa, "AA": float(class_accuracy.mean()), "Kappa": kappa, "CA": class_accuracy.tolist()}


def run_epoch(
    model: nn.Module,
    loader: data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    hsi_bands: int,
    num_classes: int,
) -> dict[str, object]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []
    context = contextlib.nullcontext() if training else torch.no_grad()
    with context:
        for batch_input, batch_target in loader:
            batch_input = batch_input.to(device, non_blocking=True)
            batch_target = batch_target.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(batch_input[:, :hsi_bands], batch_input[:, hsi_bands:])
            loss = criterion(output, batch_target)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach().item()) * batch_target.size(0)
            targets.extend(batch_target.detach().cpu().tolist())
            predictions.extend(output.detach().argmax(dim=1).cpu().tolist())
    result = metrics(targets, predictions, num_classes)
    result["loss"] = total_loss / max(1, len(targets))
    return result


def environment(device: torch.device) -> dict[str, object]:
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda": torch.version.cuda,
        "cudnn": cudnn.version(),
        "device": str(device),
    }
    if device.type == "cuda":
        result["gpu"] = torch.cuda.get_device_name(device)
        result["gpu_capability"] = torch.cuda.get_device_capability(device)
    return result


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, default=str)


def save_metrics_csv(path: Path, records: Iterable[dict[str, object]]) -> None:
    fields = [
        "epoch", "learning_rate", "train_loss", "train_OA", "train_AA", "train_Kappa",
        "test_loss", "test_OA", "test_AA", "test_Kappa",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def parameter_count(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def train_or_test(dataset_name: str, args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    config = DATASETS[dataset_name]
    patch_size = args.patch_size or config.patch_size
    batch_size = args.batch_size or config.batch_size
    learning_rate = args.learning_rate or config.learning_rate
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / f"{dataset_name}_{args.mode}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    log_path = run_dir / "stdout.log"
    with log_path.open("w") as log_handle, contextlib.redirect_stdout(Tee(sys.stdout, log_handle)):
        set_seed(args.seed)
        print(f"Starting {config.name} ({args.mode})")
        print(f"Output: {run_dir}")
        bundle = load_dataset(dataset_name, args.data_root, patch_size)
        pin_memory = device.type == "cuda"
        train_loader = make_loader(bundle, "train", batch_size, args.num_workers, pin_memory)
        test_loader = make_loader(bundle, "test", batch_size, args.num_workers, pin_memory)
        model = M2Heat(
            patch_size=patch_size,
            num_classes=bundle.num_classes,
            num_patches=[bundle.hsi_bands, bundle.auxiliary_bands],
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
        ).to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        total_params, trainable_params = parameter_count(model)
        run_config = {
            "dataset": dataset_name,
            "mode": args.mode,
            "seed": args.seed,
            "epochs": args.epochs,
            "eval_every": args.eval_every,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "patch_size": patch_size,
            "optimizer": "Adam",
            "adam_betas": [0.9, 0.999],
            "weight_decay": args.weight_decay,
            "scheduler": {"name": "StepLR", "step_size": 20, "gamma": args.gamma},
            "normalization": "bandwise min-max for HSI; auxiliary band unchanged",
            "augmentation": "none",
            "fve_initialization": "trunc_normal(std=0.02), trainable=True",
            "train_samples": len(bundle.target_train),
            "test_samples": len(bundle.target_test),
            "hsi_bands": bundle.hsi_bands,
            "auxiliary_bands": bundle.auxiliary_bands,
            "num_classes": bundle.num_classes,
            "image_size": [bundle.height, bundle.width],
            "params": total_params,
            "trainable_params": trainable_params,
            "device": str(device),
        }
        save_json(run_dir / "config.json", run_config)
        save_json(run_dir / "environment.json", environment(device))
        print(json.dumps(run_config, indent=2))

        if args.mode == "test":
            if args.checkpoint is None:
                raise ValueError("--checkpoint is required when --mode test is used")
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint.get("model_state", checkpoint))
            result = run_epoch(model, test_loader, criterion, None, device, bundle.hsi_bands, bundle.num_classes)
            print(
                f"Test: loss={result['loss']:.4f}, OA={result['OA'] * 100:.2f}, "
                f"AA={result['AA'] * 100:.2f}, Kappa={result['Kappa']:.4f}"
            )
            save_json(run_dir / "metrics.json", {"dataset": dataset_name, "test": result, "checkpoint": str(args.checkpoint)})
            return {
                "dataset": config.name,
                "mode": args.mode,
                "best_epoch": checkpoint.get("best_metrics", {}).get("epoch", "") if isinstance(checkpoint, dict) else "",
                "best_OA": result["OA"],
                "best_AA": result["AA"],
                "best_Kappa": result["Kappa"],
                "runtime_sec": 0.0,
                "params": total_params,
                "trainable_params": trainable_params,
                "run_dir": str(run_dir),
            }

        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=args.gamma)
        best: dict[str, object] = {"epoch": -1, "OA": -1.0, "AA": 0.0, "Kappa": -1.0, "CA": []}
        records: list[dict[str, object]] = []
        start = time.time()
        for epoch in range(args.epochs):
            train_result = run_epoch(model, train_loader, criterion, optimizer, device, bundle.hsi_bands, bundle.num_classes)
            scheduler.step()
            if epoch % args.eval_every != 0 and epoch != args.epochs - 1:
                continue
            test_result = run_epoch(model, test_loader, criterion, None, device, bundle.hsi_bands, bundle.num_classes)
            record = {
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": train_result["loss"],
                "train_OA": train_result["OA"],
                "train_AA": train_result["AA"],
                "train_Kappa": train_result["Kappa"],
                "test_loss": test_result["loss"],
                "test_OA": test_result["OA"],
                "test_AA": test_result["AA"],
                "test_Kappa": test_result["Kappa"],
            }
            records.append(record)
            print(
                f"Epoch {epoch + 1:03d}: train_loss={train_result['loss']:.4f}, "
                f"train_OA={train_result['OA'] * 100:.2f}, test_OA={test_result['OA'] * 100:.2f}, "
                f"test_AA={test_result['AA'] * 100:.2f}, test_Kappa={test_result['Kappa']:.4f}"
            )
            if test_result["OA"] > best["OA"]:
                best = {"epoch": epoch + 1, **test_result}
                torch.save(
                    {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                     "scheduler_state": scheduler.state_dict(), "epoch": epoch + 1,
                     "best_metrics": best, "config": run_config},
                    run_dir / "best.pt",
                )

        runtime = time.time() - start
        torch.save(
            {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
             "scheduler_state": scheduler.state_dict(), "epoch": args.epochs,
             "best_metrics": best, "config": run_config},
            run_dir / "last.pt",
        )
        save_metrics_csv(run_dir / "metrics.csv", records)
        result = {
            "dataset": config.name,
            "mode": args.mode,
            "best_epoch": best["epoch"],
            "best_OA": best["OA"],
            "best_AA": best["AA"],
            "best_Kappa": best["Kappa"],
            "runtime_sec": runtime,
            "params": total_params,
            "trainable_params": trainable_params,
            "run_dir": str(run_dir),
        }
        save_json(run_dir / "metrics.json", result | {"best": best, "final_record": records[-1] if records else {}})
        print(
            f"Finished {config.name}: best_epoch={best['epoch']}, OA={best['OA'] * 100:.2f}, "
            f"AA={best['AA'] * 100:.2f}, Kappa={best['Kappa']:.4f}, runtime={runtime:.1f}s"
        )
        return result


def save_summary(output_root: Path, results: list[dict[str, object]]) -> None:
    if not results:
        return
    fields = ["dataset", "mode", "best_epoch", "best_OA", "best_AA", "best_Kappa", "runtime_sec", "params", "trainable_params", "run_dir"]
    with (output_root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: result.get(field, "") for field in fields} for result in results)
    with (output_root / "summary.md").open("w") as handle:
        handle.write("# M2Heat Results\n\n")
        handle.write("Results are generated by `demo.py`; OA and AA are reported as percentages.\n\n")
        handle.write("| Dataset | Mode | Best epoch | OA | AA | Kappa | Runtime (s) | Run |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|---|\n")
        for result in results:
            handle.write(
                f"| {result.get('dataset', '')} | {result.get('mode', '')} | {result.get('best_epoch', '')} | "
                f"{float(result.get('best_OA', 0.0)) * 100:.2f} | {float(result.get('best_AA', 0.0)) * 100:.2f} | "
                f"{float(result.get('best_Kappa', 0.0)):.4f} | {float(result.get('runtime_sec', 0.0)):.1f} | "
                f"`{result.get('run_dir', '')}` |\n"
            )


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.eval_every < 1:
        raise ValueError("--epochs and --eval-every must be positive")
    device = select_device(args.device, args.gpu_id)
    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    results = []
    for dataset_name in datasets:
        results.append(train_or_test(dataset_name, args, device))
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    save_summary(args.output_root, results)


if __name__ == "__main__":
    main()
