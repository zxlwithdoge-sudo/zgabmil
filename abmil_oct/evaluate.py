import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from .checkpoints import load_checkpoint
from .data import OCTBags


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate frame- and subject-level predictions"
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="*")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def evaluate_fold(checkpoint_path, features_dir, device):
    checkpoint, model = load_checkpoint(checkpoint_path, device)
    config = checkpoint["args"]
    label0, label1 = config["label0"], config["label1"]
    label_names = {0: label0, 1: label1}
    needs_coordinates = getattr(model, "uses_coordinates", False)
    dataset = OCTBags(
        checkpoint["test_subjects"],
        features_dir,
        label0,
        label1,
        patch_size=config.get("patch_size"),
        require_coordinates=needs_coordinates,
    )
    subjects = defaultdict(
        lambda: {
            "true_label": None,
            "predictions": [],
            "probabilities": [],
            "slides": [],
        }
    )

    with torch.inference_mode():
        for sample in dataset.samples:
            bag = sample["bag"].to(device)
            coordinates = (
                sample["coordinates"].to(device) if needs_coordinates else None
            )
            _, probability, prediction, _ = model(bag, coordinates)
            true_label = int(sample["label"])
            pred_label = int(prediction.item())
            class1_probability = float(probability[0, 1].item())
            entry = subjects[sample["subject_id"]]
            entry["true_label"] = true_label
            entry["predictions"].append(pred_label)
            entry["probabilities"].append(class1_probability)
            entry["slides"].append(
                {
                    "case_id": sample["case_id"],
                    "true_label": label_names[true_label],
                    "pred_label": label_names[pred_label],
                    "correct": true_label == pred_label,
                    "class1_probability": class1_probability,
                }
            )

    rows = []
    for subject_id in sorted(subjects):
        entry = subjects[subject_id]
        counts = Counter(entry["predictions"])
        majority_prediction = counts.most_common(1)[0][0]
        rows.append(
            {
                "subject_id": subject_id,
                "true_label": label_names[entry["true_label"]],
                "pred_label": label_names[majority_prediction],
                "subject_correct": majority_prediction
                == entry["true_label"],
                "n_slides": len(entry["slides"]),
                "slide_accuracy": float(
                    np.mean(
                        [
                            prediction == entry["true_label"]
                            for prediction in entry["predictions"]
                        ]
                    )
                ),
                "mean_class1_probability": float(
                    np.mean(entry["probabilities"])
                ),
                "prediction_counts": dict(counts),
                "slides": entry["slides"],
            }
        )
    return rows


def main():
    args = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    checkpoints = sorted(
        args.results_dir.glob("model_checkpoint_fold_*.pt"),
        key=lambda path: int(path.stem.rsplit("_", 1)[1]),
    )
    if args.folds:
        requested = set(args.folds)
        checkpoints = [
            path
            for path in checkpoints
            if int(path.stem.rsplit("_", 1)[1]) in requested
        ]
    if not checkpoints:
        raise FileNotFoundError(
            f"No matching fold checkpoints in {args.results_dir}"
        )

    subject_rows, slide_rows, fold_summaries = [], [], []
    detailed_folds = []
    for checkpoint_path in checkpoints:
        fold = int(checkpoint_path.stem.rsplit("_", 1)[1])
        subjects = evaluate_fold(checkpoint_path, args.features_dir, device)
        accuracy = float(
            np.mean([subject["subject_correct"] for subject in subjects])
        )
        fold_summaries.append(
            {"fold": fold, "subject_accuracy": accuracy}
        )
        detailed_folds.append({"fold": fold, "subjects": subjects})

        for subject in subjects:
            subject_rows.append(
                {
                    "fold": fold,
                    "subject_id": subject["subject_id"],
                    "true_label": subject["true_label"],
                    "pred_label": subject["pred_label"],
                    "subject_correct": subject["subject_correct"],
                    "n_slides": subject["n_slides"],
                    "slide_accuracy": subject["slide_accuracy"],
                    "mean_class1_probability": subject[
                        "mean_class1_probability"
                    ],
                    "prediction_counts": json.dumps(
                        subject["prediction_counts"], sort_keys=True
                    ),
                }
            )
            for slide in subject["slides"]:
                slide_rows.append(
                    {
                        "fold": fold,
                        "subject_id": subject["subject_id"],
                        **slide,
                    }
                )

    with (args.results_dir / "subject_predictions.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "fold",
                "subject_id",
                "true_label",
                "pred_label",
                "subject_correct",
                "n_slides",
                "slide_accuracy",
                "mean_class1_probability",
                "prediction_counts",
            ],
        )
        writer.writeheader()
        writer.writerows(subject_rows)

    with (args.results_dir / "slide_predictions_evaluated.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "fold",
                "subject_id",
                "case_id",
                "true_label",
                "pred_label",
                "correct",
                "class1_probability",
            ],
        )
        writer.writeheader()
        writer.writerows(slide_rows)

    summary = {
        "device": str(device),
        "folds": fold_summaries,
        "mean_subject_accuracy": float(
            np.mean([row["subject_accuracy"] for row in fold_summaries])
        ),
        "std_subject_accuracy": float(
            np.std([row["subject_accuracy"] for row in fold_summaries])
        ),
        "details": detailed_folds,
    }
    with (args.results_dir / "subject_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(
        f"Subject accuracy: {summary['mean_subject_accuracy']:.3f} "
        f"+/- {summary['std_subject_accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()
