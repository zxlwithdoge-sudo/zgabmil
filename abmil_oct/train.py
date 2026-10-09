import argparse
import copy
import csv
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .additive import AdditiveOCTAttention
from .data import OCTBags
from .model import SPATIAL_MODES, OCTAttention
from .provenance import software_versions
from .splits import load_splits


def parse_args():
    parser = argparse.ArgumentParser(description="Train ABMIL-OCT")
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--splits-path", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--label0", required=True)
    parser.add_argument("--label1", required=True)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument(
        "--spatial-mode",
        choices=SPATIAL_MODES,
        default="none",
        help=(
            "GABMIL spatial information mixing: 'none' reproduces ABMIL, "
            "'block' mixes local windows, 'grid' mixes a dilated grid, "
            "'block_grid' applies both"
        ),
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=3,
        help="BLOCK window side length in patches (GABMIL win_size_b)",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=3,
        help="GRID window side length in patches (GABMIL win_size_g)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=112,
        help="Patch stride in pixels, used to build the patch grid",
    )
    parser.add_argument(
        "--head",
        choices=("linear", "additive"),
        default="linear",
        help=(
            "Bag predictor. 'linear' is the manuscript model: one Linear on "
            "the attention-pooled embedding. 'additive' scores every patch "
            "with a non-linear MLP and sums the scores (Javed et al., NeurIPS "
            "2022). Both decompose exactly into per-patch contributions - see "
            "abmil_oct/contributions.py - so this flag changes the model "
            "class, not the interpretability."
        ),
    )
    parser.add_argument(
        "--head-hidden",
        type=int,
        nargs="*",
        default=[256],
        help=(
            "Hidden widths of the additive patch scorer. Empty reproduces the "
            "linear head. Ignored when --head linear."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--flush-denormals",
        action="store_true",
        help=(
            "Flush subnormal floats to zero. CPU training slows down by an "
            "order of magnitude once activations drift into the subnormal "
            "range, usually after a few epochs; this avoids that. It is a "
            "numerical change, so it is off by default and is recorded in "
            "run_config.json. Irrelevant on GPU."
        ),
    )
    return parser.parse_args()


def set_seed(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def make_sampler(dataset):
    labels = [sample["label"] for sample in dataset.samples]
    counts = Counter(labels)
    weights = [len(labels) / counts[label] for label in labels]
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def safe_auc(labels, probabilities):
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probabilities))


def run_epoch(model, loader, device, loss_function, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels, predictions, probabilities, records = [], [], [], []
    needs_coordinates = getattr(model, "uses_coordinates", False)

    for batch in loader:
        bag = batch["bag"].to(device)
        label = batch["label"].view(-1).to(device)
        coordinates = (
            batch["coordinates"].to(device) if needs_coordinates else None
        )
        with torch.set_grad_enabled(training):
            logits, probability, prediction, _ = model(bag, coordinates)
            loss = loss_function(logits, label)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        true_value = int(label.item())
        predicted_value = int(prediction.item())
        class1_probability = float(probability[0, 1].item())
        total_loss += float(loss.item())
        labels.append(true_value)
        predictions.append(predicted_value)
        probabilities.append(class1_probability)
        records.append(
            {
                "case_id": batch["case_id"][0],
                "subject_id": batch["subject_id"][0],
                "true_label": true_value,
                "pred_label": predicted_value,
                "class1_probability": class1_probability,
            }
        )

    return {
        "loss": total_loss / len(loader),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "auc": safe_auc(labels, probabilities),
        "records": records,
    }


def scalar_metrics(metrics):
    return {key: metrics[key] for key in ("loss", "accuracy", "balanced_accuracy", "auc")}


def save_predictions(path, fold, split_name, records, label0, label1):
    names = {0: label0, 1: label1}
    fieldnames = [
        "fold",
        "split",
        "case_id",
        "subject_id",
        "true_label",
        "pred_label",
        "class1_probability",
        "correct",
    ]
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "fold": fold,
                    "split": split_name,
                    "case_id": record["case_id"],
                    "subject_id": record["subject_id"],
                    "true_label": names[record["true_label"]],
                    "pred_label": names[record["pred_label"]],
                    "class1_probability": record["class1_probability"],
                    "correct": record["true_label"] == record["pred_label"],
                }
            )


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.flush_denormals:
        torch.set_flush_denormal(True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    folds = load_splits(args.splits_path)
    if args.folds > len(folds):
        raise ValueError(
            f"Requested {args.folds} folds, split file has {len(folds)}"
        )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.results_dir / "slide_predictions.csv"
    if predictions_path.exists():
        predictions_path.unlink()

    config = {
        **vars(args),
        "features_dir": str(args.features_dir),
        "splits_path": str(args.splits_path),
        "results_dir": str(args.results_dir),
        "device": str(device),
        "model": ("ABMIL" if args.spatial_mode == "none" else "GABMIL")
        + ("" if args.head == "linear" else "+Additive"),
        "head": args.head,
        "head_hidden": list(args.head_hidden),
        "head_bias": False,
        "feature_dim": 1024,
        "checkpoint_criterion": "last",
        "software": software_versions(),
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
    }
    with (args.results_dir / "run_config.json").open("w") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)

    loss_function = nn.CrossEntropyLoss()
    fold_rows = []
    print(f"Using device: {device}")
    if args.spatial_mode == "none":
        print("Model: ABMIL (no spatial mixing)")
    else:
        print(
            f"Model: GABMIL, spatial_mode={args.spatial_mode}, "
            f"block_size={args.block_size}, grid_size={args.grid_size}, "
            f"patch_size={args.patch_size}"
        )
    if args.head == "additive":
        print(f"Head: additive, psi hidden widths {tuple(args.head_hidden)}")

    for fold_index, (train_subjects, test_subjects) in enumerate(
        folds[: args.folds], start=1
    ):
        print(f"\nFold {fold_index}/{args.folds}")
        needs_coordinates = args.spatial_mode != "none"
        dataset_options = {
            "patch_size": args.patch_size,
            "require_coordinates": needs_coordinates,
        }
        train_data = OCTBags(
            train_subjects,
            args.features_dir,
            args.label0,
            args.label1,
            **dataset_options,
        )
        test_data = OCTBags(
            test_subjects,
            args.features_dir,
            args.label0,
            args.label1,
            **dataset_options,
        )
        if train_data.feature_dim != 1024 or test_data.feature_dim != 1024:
            raise ValueError("The manuscript model requires 1024-D features")

        loader_options = {
            "batch_size": 1,
            "num_workers": args.num_workers,
            "pin_memory": device.type == "cuda",
        }
        train_loader = DataLoader(
            train_data,
            sampler=make_sampler(train_data),
            **loader_options,
        )
        test_loader = DataLoader(
            test_data,
            shuffle=False,
            **loader_options,
        )

        shared = dict(
            dropout=args.dropout,
            spatial_mode=args.spatial_mode,
            block_size=args.block_size,
            grid_size=args.grid_size,
        )
        if args.head == "additive":
            model = AdditiveOCTAttention(
                head_hidden=tuple(args.head_hidden), **shared
            ).to(device)
        else:
            model = OCTAttention(**shared).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
        )
        history = []
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                model, train_loader, device, loss_function, optimizer
            )
            test_metrics = run_epoch(model, test_loader, device, loss_function)
            row = {
                "epoch": epoch,
                **{
                    f"train_{key}": value
                    for key, value in scalar_metrics(train_metrics).items()
                },
                **{
                    f"test_{key}": value
                    for key, value in scalar_metrics(test_metrics).items()
                },
            }
            history.append(row)
            print(
                f"Epoch {epoch:02d}: "
                f"train acc={train_metrics['accuracy']:.3f}, "
                f"test acc={test_metrics['accuracy']:.3f}, "
                f"test AUC={test_metrics['auc']:.3f}"
            )

        checkpoint = {
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "epoch": args.epochs,
            "args": config,
            "train_subjects": train_subjects,
            "test_subjects": test_subjects,
            "train_metrics": scalar_metrics(train_metrics),
            "test_metrics": scalar_metrics(test_metrics),
        }
        torch.save(
            checkpoint,
            args.results_dir / f"model_checkpoint_fold_{fold_index}.pt",
        )

        with (
            args.results_dir / f"fold_{fold_index}_history.csv"
        ).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)

        save_predictions(
            predictions_path,
            fold_index,
            "train",
            train_metrics["records"],
            args.label0,
            args.label1,
        )
        save_predictions(
            predictions_path,
            fold_index,
            "test",
            test_metrics["records"],
            args.label0,
            args.label1,
        )
        fold_rows.append(
            {
                "fold": fold_index,
                **{
                    f"train_{key}": value
                    for key, value in scalar_metrics(train_metrics).items()
                },
                **{
                    f"test_{key}": value
                    for key, value in scalar_metrics(test_metrics).items()
                },
            }
        )

    metric_names = [
        "train_loss",
        "train_accuracy",
        "train_balanced_accuracy",
        "train_auc",
        "test_loss",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_auc",
    ]
    with (args.results_dir / "fold_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["fold", *metric_names]
        )
        writer.writeheader()
        writer.writerows(fold_rows)
        for name, function in (("mean", np.nanmean), ("std", np.nanstd)):
            writer.writerow(
                {
                    "fold": name,
                    **{
                        metric: float(
                            function([row[metric] for row in fold_rows])
                        )
                        for metric in metric_names
                    },
                }
            )

    summary = {
        "config": config,
        "folds": fold_rows,
        "mean_test_accuracy": float(
            np.nanmean([row["test_accuracy"] for row in fold_rows])
        ),
        "std_test_accuracy": float(
            np.nanstd([row["test_accuracy"] for row in fold_rows])
        ),
        "mean_test_balanced_accuracy": float(
            np.nanmean(
                [row["test_balanced_accuracy"] for row in fold_rows]
            )
        ),
        "mean_test_auc": float(
            np.nanmean([row["test_auc"] for row in fold_rows])
        ),
        "std_test_auc": float(
            np.nanstd([row["test_auc"] for row in fold_rows])
        ),
    }
    with (args.results_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved results to {args.results_dir}")


if __name__ == "__main__":
    main()
