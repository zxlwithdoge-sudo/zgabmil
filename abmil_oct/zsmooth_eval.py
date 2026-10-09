"""Zero-parameter test of z-direction attention continuity.

Motivation: consecutive OCT frames of one subject are slices through the same
tissue. On this cohort the same patch position in frame k and k+1 has cosine
similarity 0.988 (vs 0.953 for in-plane neighbours), so a lesion attended in one
frame should still be there in the next. This script tests whether exploiting
that helps, WITHOUT retraining and WITHOUT adding a single parameter.

Method: run a trained checkpoint, take the raw attention logits per frame, and
average each patch's logit with the logits of the same (row, column) position in
the +/- w neighbouring frames of the same subject. Then re-pool and re-classify.

`--windows 0` reproduces the stock model exactly and is the correctness check:
its numbers must match the checkpoint's own subject_summary.json. If they do not,
stop and fix that before reading anything else.

Usage
-----
python -m abmil_oct.zsmooth_eval \
  --results-dir results/ad_nc_gabmil/none \
  --features-dir abmil_oct/data/features_112_masked \
  --windows 0 1 2 3 5
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from .checkpoints import load_checkpoint
from .data import OCTBags


def parse_args():
    p = argparse.ArgumentParser(
        description="Inference-only test of z-smoothed attention (no retraining)"
    )
    p.add_argument("--results-dir", type=Path, required=True,
                   help="folder holding model_checkpoint_fold_*.pt")
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--windows", type=int, nargs="+", default=[0, 1, 2, 3, 5],
                   help="half-widths w; frame k averages over k-w..k+w. 0 = stock model")
    p.add_argument("--folds", type=int, nargs="*")
    p.add_argument("--output", type=Path,
                   help="CSV to write (default: <results-dir>/zsmooth_results.csv)")
    p.add_argument("--grid-origin", choices=["absolute", "frame"], default="absolute",
                   help="'absolute' puts every frame on one lattice (correct for "
                        "cross-frame matching). 'frame' reproduces the original "
                        "per-frame-normalised behaviour, which misaligns 4-13%% of "
                        "adjacent frame pairs by a patch; kept only to reproduce "
                        "the 2026-09-29 run.")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def frame_index(case_id):
    """AD_20832_137 -> 137. Frames are consecutive slices through one volume."""
    return int(case_id.rsplit("_", 1)[1])


def safe_auc(labels, probabilities):
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probabilities))


@torch.inference_mode()
def encode_fold(model, dataset, device):
    """One forward pass per frame; keep embeddings, attention logits, positions."""
    per_subject = defaultdict(dict)
    for sample in dataset.samples:
        bag = sample["bag"].to(device)
        embedded = model.embedding(bag)                       # [N, 512]
        logits = model.attention(embedded).reshape(-1)        # [N]  raw attention
        per_subject[sample["subject_id"]][frame_index(sample["case_id"])] = {
            "embedded": embedded,
            "logit": logits,
            "coords": sample["coordinates"].cpu().numpy(),    # [N, 2] grid indices
            "label": int(sample["label"]),
            "case_id": sample["case_id"],
        }
    return per_subject


def smooth_logits(frames, k, w, stats=None):
    """Average frame k's attention logits with the same (row, col) in k-w..k+w.

    Positions missing from a neighbouring frame simply contribute nothing, so a
    patch at the edge of the tissue mask is never dragged toward zero by absence.
    """
    current = frames[k]
    if w == 0:
        return current["logit"]

    neighbours_found = 0
    total = current["logit"].clone()
    count = torch.ones_like(total)
    for offset in range(-w, w + 1):
        if offset == 0 or (k + offset) not in frames:
            continue
        neighbours_found += 1
        neighbour = frames[k + offset]
        lookup = {tuple(rc): i for i, rc in enumerate(neighbour["coords"])}
        rows = [(i, lookup[tuple(rc)]) for i, rc in enumerate(current["coords"])
                if tuple(rc) in lookup]
        if not rows:
            continue
        here = torch.tensor([i for i, _ in rows], device=total.device)
        there = torch.tensor([j for _, j in rows], device=total.device)
        total.index_add_(0, here, neighbour["logit"][there])
        count.index_add_(0, here, torch.ones_like(there, dtype=total.dtype))
    smoothed = total / count
    if stats is not None:
        stats["neighbours"].append(neighbours_found)
        a0 = torch.softmax(current["logit"], dim=0)
        a1 = torch.softmax(smoothed, dim=0)
        stats["attention_shift"].append(float((a1 - a0).abs().sum()))
    return smoothed


@torch.inference_mode()
def evaluate_window(model, per_subject, w):
    labels, preds, probs = [], [], []
    subject_votes = defaultdict(lambda: {"true": None, "pred": []})
    stats = {"neighbours": [], "attention_shift": []}

    for subject_id, frames in per_subject.items():
        for k in sorted(frames):
            entry = frames[k]
            attention = torch.softmax(smooth_logits(frames, k, w, stats), dim=0)
            pooled = attention.unsqueeze(0) @ entry["embedded"]      # [1, 512]
            logits = model.classifier(pooled)
            probability = torch.softmax(logits, dim=1)
            prediction = int(logits.argmax(dim=1).item())

            labels.append(entry["label"])
            preds.append(prediction)
            probs.append(float(probability[0, 1].item()))
            subject_votes[subject_id]["true"] = entry["label"]
            subject_votes[subject_id]["pred"].append(prediction)

    subject_correct = []
    for v in subject_votes.values():
        majority = Counter(v["pred"]).most_common(1)[0][0]
        subject_correct.append(majority == v["true"])

    return {
        "frame_accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "frame_auc": safe_auc(labels, probs),
        "subject_accuracy": float(np.mean(subject_correct)),
        "n_subjects": len(subject_correct),
        # diagnostics: separate "smoothing did not help" from "smoothing did nothing"
        "mean_neighbours_used": float(np.mean(stats["neighbours"])) if stats["neighbours"] else 0.0,
        "mean_attention_shift": float(np.mean(stats["attention_shift"])) if stats["attention_shift"] else 0.0,
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    checkpoints = sorted(args.results_dir.glob("model_checkpoint_fold_*.pt"),
                         key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if args.folds:
        wanted = set(args.folds)
        checkpoints = [p for p in checkpoints
                       if int(p.stem.rsplit("_", 1)[1]) in wanted]
    if not checkpoints:
        raise FileNotFoundError(f"no fold checkpoints in {args.results_dir}")

    rows = []
    for path in checkpoints:
        fold = int(path.stem.rsplit("_", 1)[1])
        checkpoint, model = load_checkpoint(path, device)
        config = checkpoint["args"]
        if config.get("spatial_mode", "none") != "none":
            print(f"note: fold {fold} is spatial_mode="
                  f"{config['spatial_mode']}; z-smoothing is applied on top of it")
        dataset = OCTBags(
            checkpoint["test_subjects"], args.features_dir,
            config["label0"], config["label1"],
            patch_size=config.get("patch_size", 112),
            require_coordinates=True,
            grid_origin=args.grid_origin,
        )
        per_subject = encode_fold(model, dataset, device)
        for w in args.windows:
            metrics = evaluate_window(model, per_subject, w)
            rows.append({"fold": fold, "window": w, **metrics})
            print(f"fold {fold:2d}  w={w:<2d} "
                  f"frame acc {metrics['frame_accuracy']:.3f}  "
                  f"bal {metrics['balanced_accuracy']:.3f}  "
                  f"AUC {metrics['frame_auc']:.3f}  "
                  f"subject {metrics['subject_accuracy']:.3f}"
                  + ("" if w == 0 else
                     f"   [neighbours/frame {metrics['mean_neighbours_used']:.1f}/{2 * w}, "
                     f"attention shift {metrics['mean_attention_shift']:.4f}]"), flush=True)

    out = args.output or (args.results_dir / "zsmooth_results.csv")
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'window':>7s} {'frame acc':>17s} {'balanced':>17s} {'frame AUC':>17s} {'subject acc':>17s}")
    baseline = {}
    for w in args.windows:
        sel = [r for r in rows if r["window"] == w]
        line = f"{w:7d} "
        for key in ("frame_accuracy", "balanced_accuracy", "frame_auc", "subject_accuracy"):
            a = np.array([r[key] for r in sel], dtype=float)
            if w == args.windows[0]:
                baseline[key] = a
            delta = f" ({np.nanmean(a) - np.nanmean(baseline[key]):+.3f})" if w != args.windows[0] else ""
            line += f"  {np.nanmean(a):.3f}+/-{np.nanstd(a):.3f}{delta:>9s}"
        print(line)
    print(f"\nwrote {out}")
    if args.windows and args.windows[0] == 0:
        print("\nCHECK 1: the w=0 row must match this checkpoint's subject_summary.json.")
    moved = [r for r in rows if r["window"] != 0]
    if moved:
        shift = float(np.mean([r["mean_attention_shift"] for r in moved]))
        used = float(np.mean([r["mean_neighbours_used"] for r in moved]))
        print(f"CHECK 2: mean attention shift {shift:.4f}, mean neighbours used {used:.1f}. "
              "A shift near 0 means smoothing did NOTHING (frames of a subject are not "
              "consecutively numbered, or grids do not align) - that is a data problem, "
              "not a negative result.")


if __name__ == "__main__":
    main()
