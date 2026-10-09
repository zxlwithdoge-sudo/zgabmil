"""Signed per-patch class contributions, and the evidence they beat attention.

Attention answers "did the model look here?". It is one unsigned scalar per
patch, shared across classes, and it sits behind a classifier, so its magnitude
has no fixed relationship to the prediction. A contribution answers "what did
this patch add to the AD logit, and to the NC logit?" - signed, per class, and
summing exactly to the bag logits.

That identity is the point::

    c_i = alpha_i * head(h_i)            sum_i c_i == logits    (exactly)

It holds for both heads in this project:

* ``model.OCTAttention`` - ``head`` is the ``nn.Linear`` classifier. Because
  ``sum_i alpha_i == 1``, folding the bias into every patch as
  ``alpha_i * (W h_i + b)`` reproduces the logits with no arbitrary split. The
  manuscript model therefore already supports exact attribution, with no
  retraining and no change in predictions.
* ``additive.AdditiveOCTAttention`` - ``head`` is the non-linear ``psi``.

So the contribution map is available today, on existing checkpoints. The
additive head is a separate question about model class, not a prerequisite for
these figures.

What this module produces
-------------------------
``main`` runs a fold checkpoint over its test subjects and writes:

``patch_contributions.csv``
    one row per patch: attention logit, per-class contribution, the signed
    margin, ``percent_of_frame_evidence`` (the margin as a signed share of the
    frame's total evidence, so the absolute values sum to 100), and the
    within-frame percentile rank of attention and of |contribution|.
``faithfulness.png``
    the check from Javed et al. (2022), Fig. 3, on this cohort: summed
    contributions against the frame logit (must be the identity line, because
    it is an identity) beside an attention summary against the same logit. If
    the right panel is a cloud, attention is not tracking the prediction.
``divergence.json``
    how far the two maps disagree: rank correlation, and the fraction of
    above-median-attention patches whose class contribution has the opposite
    sign to the frame's prediction. Those patches are the ones an attention
    heatmap shows as important and a contribution map shows as evidence
    against.

Rendering note: attention is a magnitude, so it takes a sequential ramp.
A contribution is a polarity, so it takes a **diverging** map with a neutral
midpoint pinned at zero and symmetric limits - never a rainbow, which invents
a midpoint where the data has none.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from .checkpoints import load_checkpoint
from .data import OCTBags


def _patch_head(model):
    """Return a callable mapping ``[N, D]`` embeddings to ``[N, C]`` scores."""
    classifier = model.classifier
    psi = getattr(classifier, "psi", None)
    return psi if psi is not None else classifier


@torch.inference_mode()
def patch_contributions(model, bag, coordinates=None):
    """Signed ``[N, num_classes]`` contributions plus the attention weights.

    Returns ``(contributions, attention, logits)``. ``contributions.sum(0)``
    equals ``logits`` to floating-point tolerance for both supported heads;
    ``tests/test_contributions.py`` asserts it.
    """
    if bag.ndim == 3:
        bag = bag.squeeze(0)
    if coordinates is not None and coordinates.ndim == 3:
        coordinates = coordinates.squeeze(0)

    mixer = getattr(model, "spatial_mixer", None)
    if mixer is not None:
        if coordinates is None:
            raise ValueError(
                "this checkpoint uses spatial mixing; pass coordinates"
            )
        bag = mixer(bag, coordinates)

    embedded = model.embedding(bag)
    raw_attention = model.attention(embedded).transpose(1, 0)
    attention = torch.softmax(raw_attention, dim=1).reshape(-1, 1)
    scores = _patch_head(model)(embedded)
    contributions = attention * scores
    return contributions, raw_attention.reshape(-1), contributions.sum(
        dim=0, keepdim=True
    )


def margin(contributions, positive_class=1):
    """Evidence for ``positive_class`` over the rest, signed."""
    c = np.asarray(contributions, dtype=float)
    other = np.delete(c, positive_class, axis=1).max(axis=1)
    return c[:, positive_class] - other


def _percentile(values):
    from scipy.stats import rankdata

    v = np.asarray(values, dtype=float).reshape(-1)
    if v.size == 0:
        return v
    return rankdata(v, method="average") / v.size


def divergence(attention, contribution_margin, predicted_class):
    """How much the attention map and the contribution map disagree.

    ``opposing_fraction`` is the share of above-median-attention patches whose
    contribution points away from the frame's own prediction. An attention
    heatmap marks those patches important; the contribution map marks them as
    evidence against. They are the concrete form of the paper's "benign mimic"
    argument.
    """
    a = np.asarray(attention, dtype=float).reshape(-1)
    m = np.asarray(contribution_margin, dtype=float).reshape(-1)
    if a.size < 3:
        return None
    attended = a > np.median(a)
    signed = m if predicted_class == 1 else -m
    rho_signed = spearmanr(a, signed).statistic
    rho_absolute = spearmanr(a, np.abs(m)).statistic
    return {
        "n_patches": int(a.size),
        "n_attended": int(attended.sum()),
        "opposing_fraction": float((signed[attended] < 0).mean())
        if attended.any()
        else float("nan"),
        "spearman_attention_vs_signed": float(rho_signed),
        "spearman_attention_vs_absolute": float(rho_absolute),
    }


# --- rendering ---------------------------------------------------------------

def save_contribution_overlay(path, image, patches, coordinates, values, title,
                              label="Contribution to predicted class"):
    """Diverging overlay, neutral at zero, limits symmetric about zero.

    Red is excitatory, blue inhibitory, grey is "this patch said nothing" -
    which a sequential or rainbow map cannot express.
    """
    import matplotlib.pyplot as plt

    canvas = np.zeros_like(image, dtype=np.float32)
    heat = np.full_like(image, np.nan, dtype=np.float32)
    for patch, (row, column), value in zip(patches, coordinates, values):
        height, width = patch.shape[:2]
        canvas[row : row + height, column : column + width] = patch
        heat[row : row + height, column : column + width] = value

    limit = float(np.nanmax(np.abs(heat))) or 1.0
    figure, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Normalized OCT frame")
    axes[1].imshow(canvas, cmap="gray", vmin=0, vmax=1)
    overlay = axes[1].imshow(
        heat, cmap="RdBu_r", vmin=-limit, vmax=limit, alpha=0.55
    )
    axes[1].set_title(label)
    for axis in axes:
        axis.axis("off")
    bar = figure.colorbar(overlay, ax=axes[1], fraction=0.046, pad=0.04)
    bar.set_label("inhibitory          0          excitatory")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_faithfulness(path, logits, summed, attention_summary):
    """Two scatters: contributions track the logit by construction; attention?"""
    import matplotlib.pyplot as plt

    logits = np.asarray(logits, dtype=float)
    summed = np.asarray(summed, dtype=float)
    attention_summary = np.asarray(attention_summary, dtype=float)

    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(True, alpha=0.18, linewidth=0.6)
        axis.set_axisbelow(True)

    axes[0].scatter(logits, summed, s=14, color="#3C6E9F",
                    edgecolor="white", linewidth=0.4)
    span = [min(logits.min(), summed.min()), max(logits.max(), summed.max())]
    axes[0].plot(span, span, color="#8A8A8A", linewidth=1.2, zorder=0)
    r0 = np.corrcoef(logits, summed)[0, 1]
    axes[0].set_title(f"Summed contributions   r = {r0:.4f}", fontsize=10)
    axes[0].set_xlabel("Frame logit margin")
    axes[0].set_ylabel("Sum of patch contributions")

    axes[1].scatter(logits, attention_summary, s=14, color="#B5651D",
                    edgecolor="white", linewidth=0.4)
    r1 = np.corrcoef(logits, attention_summary)[0, 1]
    axes[1].set_title(f"Top-10% attention (median)   r = {r1:.4f}", fontsize=10)
    axes[1].set_xlabel("Frame logit margin")
    axes[1].set_ylabel("Median attention logit")

    figure.suptitle(
        "Does the heatmap track the prediction?", fontsize=12, y=1.02
    )
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return float(r0), float(r1)


# --- CLI ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Per-patch signed class contributions from a fold checkpoint"
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--fold-split", choices=("train", "test", "all"),
                   default="test")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main():
    import csv

    args = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    checkpoint, model = load_checkpoint(args.checkpoint, device)
    config = checkpoint["args"]
    subjects = {
        "train": checkpoint["train_subjects"],
        "test": checkpoint["test_subjects"],
        "all": checkpoint["train_subjects"] + checkpoint["test_subjects"],
    }[args.fold_split]

    dataset = OCTBags(
        subjects, args.features_dir, config["label0"], config["label1"],
        patch_size=config.get("patch_size", 112),
        require_coordinates=bool(config.get("spatial_mode", "none") != "none"),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    frame_logits, frame_sums, frame_attention = [], [], []
    per_frame_divergence = []

    for sample in dataset.samples:
        bag = sample["bag"].to(device)
        coordinates = sample.get("coordinates")
        if coordinates is not None:
            coordinates = coordinates.to(device)
        contributions, raw_attention, logits = patch_contributions(
            model, bag, coordinates
        )
        c = contributions.cpu().numpy()
        a = raw_attention.cpu().numpy()
        logit_margin = float(logits[0, 1] - logits[0, 0])
        predicted = int(logits.argmax(dim=1).item())
        m = margin(c)

        frame_logits.append(logit_margin)
        frame_sums.append(float(m.sum()))
        cut = np.quantile(a, 0.90)
        frame_attention.append(float(np.median(a[a >= cut])))

        stats = divergence(a, m, predicted)
        if stats:
            per_frame_divergence.append(stats)

        # Signed share of the frame's total evidence. |percent| sums to 100 by
        # construction, so a patch's number reads as "this patch is x% of why
        # the model decided", on the same scale in every frame - unlike the raw
        # value, whose range moves with how confident the frame happens to be.
        percent = 100.0 * m / (np.abs(m).sum() or 1.0)
        a_rank, m_rank = _percentile(a), _percentile(np.abs(m))
        coords = (
            sample["coordinates"].cpu().numpy()
            if sample.get("coordinates") is not None
            else np.full((len(a), 2), -1)
        )
        for i in range(len(a)):
            rows.append({
                "case_id": sample["case_id"],
                "subject_id": sample["subject_id"],
                "label": int(sample["label"]),
                "predicted": predicted,
                "patch": i,
                "row": int(coords[i, 0]),
                "column": int(coords[i, 1]),
                "attention_logit": float(a[i]),
                "contribution_class0": float(c[i, 0]),
                "contribution_class1": float(c[i, 1]),
                "margin": float(m[i]),
                "percent_of_frame_evidence": float(percent[i]),
                "attention_percentile": float(a_rank[i]),
                "abs_contribution_percentile": float(m_rank[i]),
            })

    csv_path = args.output_dir / "patch_contributions.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    r_contribution, r_attention = save_faithfulness(
        args.output_dir / "faithfulness.png",
        frame_logits, frame_sums, frame_attention,
    )

    summary = {
        "n_frames": len(frame_logits),
        "n_patches": len(rows),
        "faithfulness_r_contributions": r_contribution,
        "faithfulness_r_attention": r_attention,
        "mean_opposing_fraction": float(
            np.nanmean([d["opposing_fraction"] for d in per_frame_divergence])
        ),
        "mean_spearman_attention_vs_signed": float(
            np.nanmean([d["spearman_attention_vs_signed"]
                        for d in per_frame_divergence])
        ),
        "mean_spearman_attention_vs_absolute": float(
            np.nanmean([d["spearman_attention_vs_absolute"]
                        for d in per_frame_divergence])
        ),
    }
    with (args.output_dir / "divergence.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {csv_path}")
    print("CHECK: faithfulness_r_contributions must be 1.0000 - it is an "
          "identity, so anything else means the decomposition is wrong.")


if __name__ == "__main__":
    main()
