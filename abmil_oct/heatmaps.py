import argparse
import csv
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
from scipy.stats import rankdata

from .checkpoints import load_checkpoint
from .data import OCTBags
from .preprocess import normalize_image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create patch-attention overlays from a fold checkpoint"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--patches-dir", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--fold-split", choices=("train", "test", "all"), default="test"
    )
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def percentile_attention(raw_attention):
    scores = np.asarray(raw_attention).reshape(-1)
    return rankdata(scores, method="average") / len(scores)


def selected_subjects(checkpoint, split):
    if split == "train":
        return checkpoint["train_subjects"]
    if split == "all":
        return checkpoint["train_subjects"] + checkpoint["test_subjects"]
    return checkpoint["test_subjects"]


def save_overlay(
    path, image, patches, coordinates, scores, title, attention_label="ABMIL attention"
):
    canvas = np.zeros_like(image, dtype=np.float32)
    heatmap = np.full_like(image, np.nan, dtype=np.float32)
    for patch, (row, column), score in zip(patches, coordinates, scores):
        height, width = patch.shape[:2]
        canvas[row : row + height, column : column + width] = patch
        heatmap[row : row + height, column : column + width] = score

    figure, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Normalized OCT frame")
    axes[1].imshow(canvas, cmap="gray", vmin=0, vmax=1)
    overlay = axes[1].imshow(
        heatmap, cmap="turbo", vmin=0, vmax=1, alpha=0.45
    )
    axes[1].set_title(attention_label)
    for axis in axes:
        axis.axis("off")
    figure.colorbar(
        overlay,
        ax=axes[1],
        fraction=0.046,
        pad=0.04,
        label="Within-frame percentile",
    )
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_top_patches(output_dir, case_id, patches, coordinates, raw_scores, top_k):
    count = min(top_k, len(patches), len(raw_scores))
    if count <= 0:
        return
    indices = np.argsort(raw_scores)[-count:][::-1]
    target = output_dir / f"{case_id}_top_patches"
    target.mkdir(parents=True, exist_ok=True)
    with (target / "metadata.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["rank", "patch_index", "raw_attention", "row", "column"]
        )
        for rank, index in enumerate(indices, start=1):
            row, column = coordinates[index]
            writer.writerow(
                [rank, int(index), float(raw_scores[index]), row, column]
            )
            plt.imsave(
                target / f"rank_{rank:02d}_patch_{index:03d}.png",
                patches[index],
                cmap="gray",
                vmin=0,
                vmax=1,
            )


def main():
    args = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    checkpoint, model = load_checkpoint(args.checkpoint, device)
    config = checkpoint["args"]
    subjects = selected_subjects(checkpoint, args.fold_split)
    if args.subjects:
        requested = set(args.subjects)
        subjects = [subject for subject in subjects if subject in requested]
    if not subjects:
        raise ValueError("No subjects selected")

    needs_coordinates = getattr(model, "uses_coordinates", False)
    dataset = OCTBags(
        subjects,
        args.features_dir,
        config["label0"],
        config["label1"],
        patch_size=config.get("patch_size"),
        require_coordinates=needs_coordinates,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spatial_mode = config.get("spatial_mode", "none")
    attention_label = (
        "ABMIL attention"
        if spatial_mode == "none"
        else f"GABMIL attention ({spatial_mode})"
    )

    saved = 0
    with torch.inference_mode():
        for index, sample in enumerate(dataset.samples):
            if args.stride > 1 and index % args.stride:
                continue
            frame_path = args.frames_dir / f"{sample['case_id']}.mat"
            patch_path = args.patches_dir / f"{sample['case_id']}.pkl"
            if not frame_path.exists() or not patch_path.exists():
                print(f"Missing frame or patches for {sample['case_id']}")
                continue

            coordinates = (
                sample["coordinates"].to(device) if needs_coordinates else None
            )
            _, _, _, raw_attention = model(
                sample["bag"].to(device), coordinates
            )
            raw_scores = raw_attention.cpu().numpy().reshape(-1)
            with patch_path.open("rb") as handle:
                patch_data = pickle.load(handle)
            if len(raw_scores) != len(patch_data["patches"]):
                raise ValueError(
                    f"Attention/patch mismatch for {sample['case_id']}"
                )

            frame = sio.loadmat(frame_path)
            if "frame" not in frame:
                raise KeyError(f"'frame' missing from {frame_path}")
            image = normalize_image(np.asarray(frame["frame"]).squeeze())
            save_overlay(
                args.output_dir / f"{sample['case_id']}.png",
                image,
                patch_data["patches"],
                patch_data["coordinates"],
                percentile_attention(raw_scores),
                sample["case_id"],
                attention_label=attention_label,
            )
            save_top_patches(
                args.output_dir,
                sample["case_id"],
                patch_data["patches"],
                patch_data["coordinates"],
                raw_scores,
                args.top_k,
            )
            saved += 1
            if args.limit is not None and saved >= args.limit:
                break
    print(f"Saved {saved} attention maps to {args.output_dir}")


if __name__ == "__main__":
    main()
