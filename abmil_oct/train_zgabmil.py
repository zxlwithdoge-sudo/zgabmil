"""Train GABMIL with the added z-axis feature mixing.

Deliberately mirrors ``abmil_oct.train``: same folds, same per-frame bag, same
per-frame cross-entropy, same last-epoch checkpoint, same subject majority vote.
The only difference is that the spatial mixer also mixes along z, so a run here
lines up row-for-row with the GABMIL ablation table.

Arms worth running, given what is already known on this cohort (in-plane mixing
hurt with a dose-response; inference-only z smoothing helped, +0.022 frame AUC):

    --spatial-mode none  --z-window 1   z mixing alone, minimal context
    --spatial-mode none  --z-window 3   z mixing alone
    --spatial-mode none  --z-window 5   z mixing alone, the smoother's plateau
    --spatial-mode block --z-window 3   in-plane BLOCK plus z
    --spatial-mode grid  --z-window 3   in-plane GRID plus z

`--z-window 0` with an in-plane mode is the original GABMIL and should
reproduce it; use `abmil_oct.train` for that rather than this script.

Cost: each step embeds 2*z_window+1 frames instead of 1. The model is small so
this is data movement rather than arithmetic, but a w=5 epoch moves roughly 11x
the bytes of a baseline epoch.

Usage
-----
python -m abmil_oct.train_zgabmil \
  --features-dir abmil_oct/data/features_112_masked \
  --splits abmil_oct/splits/AD_NC_folds.json \
  --label0 NC --label1 AD \
  --results-dir results/ad_nc_zgabmil/z3 \
  --spatial-mode none --z-window 3
"""

import argparse
import copy
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from .train import make_sampler, safe_auc, save_predictions, set_seed
from .volume_data import OCTContextBags, collate_single
from .zgabmil import GABMILZ


def parse_args():
    p = argparse.ArgumentParser(
        description="Train GABMIL with z-axis feature mixing"
    )
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--label0", default="NC")
    p.add_argument("--label1", default="AD")

    p.add_argument("--spatial-mode", default="none",
                   choices=["none", "block", "grid", "block_grid"],
                   help="in-plane GABMIL mixing; 'none' runs the z stage alone")
    p.add_argument("--block-size", type=int, default=3)
    p.add_argument("--grid-size", type=int, default=3)
    p.add_argument("--z-window", type=int, default=1,
                   help="half-width along the frame axis; the mixer sees 2w+1 frames")

    p.add_argument("--patch-size", type=int, default=112)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--folds", type=int, nargs="*")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--flush-denormals", action="store_true",
                   help="see README; ~25x faster CPU training, changes arithmetic")
    return p.parse_args()


def run_epoch(model, loader, device, loss_function, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels, predictions, probabilities, records = [], [], [], []

    for batch in loader:
        bag = batch["bag"].to(device)
        coordinates = batch["coordinates"].to(device)
        frames = batch["frames"].to(device)
        label = batch["label"].view(-1).to(device)

        with torch.set_grad_enabled(training):
            logits, probability, prediction, _ = model(
                bag, coordinates, frames, batch["target_frame"]
            )
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
                "case_id": batch["case_id"],
                "subject_id": batch["subject_id"],
                "true_label": true_value,
                "pred_label": predicted_value,
                "class1_probability": class1_probability,
            }
        )

    return {
        "loss": total_loss / len(loader),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "auc": safe_auc(labels, probabilities),
        "records": records,
    }


def scalar_metrics(metrics):
    return {k: metrics[k] for k in ("loss", "accuracy", "balanced_accuracy", "auc")}


def subject_vote(records, label_names):
    """Majority vote per subject, matching ``abmil_oct.evaluate`` exactly."""
    subjects = defaultdict(lambda: {"true": None, "preds": [], "probs": []})
    for record in records:
        entry = subjects[record["subject_id"]]
        entry["true"] = record["true_label"]
        entry["preds"].append(record["pred_label"])
        entry["probs"].append(record["class1_probability"])

    rows = []
    for subject_id in sorted(subjects):
        entry = subjects[subject_id]
        counts = Counter(entry["preds"])
        majority = counts.most_common(1)[0][0]
        rows.append(
            {
                "subject_id": subject_id,
                "true_label": label_names[entry["true"]],
                "pred_label": label_names[majority],
                "subject_correct": majority == entry["true"],
                "n_slides": len(entry["preds"]),
                "slide_accuracy": float(
                    np.mean([p == entry["true"] for p in entry["preds"]])
                ),
                "mean_class1_probability": float(np.mean(entry["probs"])),
                "prediction_counts": dict(counts),
            }
        )
    return rows


def main():
    args = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    if args.flush_denormals:
        torch.set_flush_denormal(True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    label_names = {0: args.label0, 1: args.label1}

    folds = json.loads(args.splits.read_text())["folds"]
    wanted = set(args.folds) if args.folds else None

    config = {
        "model": "GABMIL-Z",
        "spatial_mode": args.spatial_mode,
        "block_size": args.block_size,
        "grid_size": args.grid_size,
        "z_window": args.z_window,
        "patch_size": args.patch_size,
        "feature_dim": 1024,
        "dropout": args.dropout,
        "label0": args.label0,
        "label1": args.label1,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "checkpoint_criterion": "last",
        "grid_origin": "absolute",
    }
    (args.results_dir / "run_config.json").write_text(json.dumps(config, indent=2))

    rows, fold_summaries, all_subject_rows = [], [], []
    for fold_index, fold in enumerate(folds, start=1):
        if wanted and fold_index not in wanted:
            continue
        set_seed(args.seed + fold_index)

        train_data = OCTContextBags(
            fold["train"], args.features_dir, args.label0, args.label1,
            patch_size=args.patch_size, z_window=args.z_window,
        )
        test_data = OCTContextBags(
            fold["test"], args.features_dir, args.label0, args.label1,
            patch_size=args.patch_size, z_window=args.z_window,
        )
        for name, dataset in (("train", train_data), ("test", test_data)):
            if dataset.subjects_with_gaps:
                print(f"  note: non-consecutive frame numbering in {name} subjects "
                      f"{dataset.subjects_with_gaps}; z mixing will treat slices "
                      "across those gaps as adjacent")

        train_loader = DataLoader(
            train_data, batch_size=1, sampler=make_sampler(train_data),
            collate_fn=collate_single,
        )
        test_loader = DataLoader(
            test_data, batch_size=1, shuffle=False, collate_fn=collate_single
        )

        model = GABMILZ(
            dropout=args.dropout, spatial_mode=args.spatial_mode,
            block_size=args.block_size, grid_size=args.grid_size,
            z_window=args.z_window,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate,
            weight_decay=args.weight_decay, betas=(0.9, 0.999),
        )
        loss_function = nn.CrossEntropyLoss()

        history = []
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                model, train_loader, device, loss_function, optimizer
            )
            test_metrics = run_epoch(model, test_loader, device, loss_function)
            history.append(
                {
                    "epoch": epoch,
                    **{f"train_{k}": v for k, v in scalar_metrics(train_metrics).items()},
                    **{f"test_{k}": v for k, v in scalar_metrics(test_metrics).items()},
                }
            )
            print(
                f"fold {fold_index:2d} epoch {epoch:2d}  "
                f"train loss {train_metrics['loss']:.3f}  "
                f"test acc {test_metrics['accuracy']:.3f}  "
                f"test AUC {test_metrics['auc']:.3f}",
                flush=True,
            )

        subject_rows = subject_vote(test_metrics["records"], label_names)
        accuracy = float(np.mean([r["subject_correct"] for r in subject_rows]))
        fold_summaries.append({"fold": fold_index, "subject_accuracy": accuracy})
        all_subject_rows.extend({"fold": fold_index, **r} for r in subject_rows)
        print(f"fold {fold_index:2d} subject accuracy {accuracy:.3f}", flush=True)

        torch.save(
            {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "epoch": args.epochs,
                "args": config,
                "train_subjects": fold["train"],
                "test_subjects": fold["test"],
                "train_metrics": scalar_metrics(train_metrics),
                "test_metrics": scalar_metrics(test_metrics),
            },
            args.results_dir / f"model_checkpoint_fold_{fold_index}.pt",
        )
        with (args.results_dir / f"fold_{fold_index}_history.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        for split_name, metrics in (("train", train_metrics), ("test", test_metrics)):
            save_predictions(
                args.results_dir / "slide_predictions.csv",
                fold_index, split_name, metrics["records"],
                args.label0, args.label1,
            )

        rows.append(
            {
                "fold": fold_index,
                "subject_accuracy": accuracy,
                **{f"train_{k}": v for k, v in scalar_metrics(train_metrics).items()},
                **{f"test_{k}": v for k, v in scalar_metrics(test_metrics).items()},
            }
        )

    with (args.results_dir / "fold_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (args.results_dir / "subject_predictions.csv").open("w", newline="") as handle:
        fieldnames = ["fold", "subject_id", "true_label", "pred_label",
                      "subject_correct", "n_slides", "slide_accuracy",
                      "mean_class1_probability", "prediction_counts"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_subject_rows:
            writer.writerow(
                {**row, "prediction_counts": json.dumps(row["prediction_counts"],
                                                        sort_keys=True)}
            )

    # Same shape as evaluate.py writes, so the two are directly comparable.
    summary = {
        "folds": fold_summaries,
        "mean_subject_accuracy": float(
            np.mean([r["subject_accuracy"] for r in fold_summaries])
        ),
        "std_subject_accuracy": float(
            np.std([r["subject_accuracy"] for r in fold_summaries])
        ),
        "mean_frame_auc": float(np.nanmean([r["test_auc"] for r in rows])),
        "std_frame_auc": float(np.nanstd([r["test_auc"] for r in rows])),
        "mean_balanced_accuracy": float(
            np.mean([r["test_balanced_accuracy"] for r in rows])
        ),
        "config": config,
    }
    (args.results_dir / "subject_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print(f"\nFrame AUC:         {summary['mean_frame_auc']:.3f} "
          f"+/- {summary['std_frame_auc']:.3f}   (baseline 0.758 +/- 0.184)")
    print(f"Balanced accuracy: {summary['mean_balanced_accuracy']:.3f}"
          f"                (baseline 0.686 +/- 0.150)")
    print(f"Subject accuracy:  {summary['mean_subject_accuracy']:.3f} "
          f"+/- {summary['std_subject_accuracy']:.3f}   (baseline 0.725 +/- 0.236; "
          "this metric moves in steps of 0.025 over 40 decisions)")


if __name__ == "__main__":
    main()
