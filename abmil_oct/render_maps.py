"""Side-by-side attention and contribution overlays for selected frames.

Two maps of the same frame, from the same checkpoint:

``attention``
    the softmax weight, one unsigned scalar per patch. A magnitude, so it takes
    a sequential ramp. It says the model looked here; it cannot say for which
    class, or whether the patch argued for or against.

``contribution``
    ``c_i = alpha_i * head(h_i)``, signed and per class, summing exactly to the
    frame logits (see ``contributions.py``). A polarity, so it takes a
    diverging map with a neutral midpoint pinned at zero and symmetric limits -
    red excitatory, blue inhibitory, grey "said nothing".

Both panels are drawn on the same patch canvas, so the only difference between
them is what is being shown. The frame image itself is rebuilt from the patch
pickles, so no .mat frames are needed.

Usage
-----
python -m abmil_oct.render_maps \
  --checkpoint results/ad_nc_gabmil/none/model_checkpoint_fold_7.pt \
  --features-dir abmil_oct/data/features_112_masked \
  --patches-dir abmil_oct/data/patches_112_masked \
  --cases AD_20832_85 NC_21499_136 \
  --output-dir results/maps_fold7
"""

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .checkpoints import load_checkpoint
from .contributions import margin, patch_contributions
from .data import OCTBags, subject_from_stem


def parse_args():
    p = argparse.ArgumentParser(
        description="Attention vs contribution overlays for chosen frames"
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--patches-dir", type=Path, required=True)
    p.add_argument("--frames-dir", type=Path, default=None,
                   help="optional: .mat frames, as heatmaps.py uses. Given, "
                        "the background is the real frame including the tissue "
                        "the mask dropped; otherwise it is rebuilt from the "
                        "patches, which shows only what the model saw.")
    p.add_argument("--output-dir", type=Path, required=True)
    # Frame selection, same interface as heatmaps.py. Omit --cases to render
    # the whole split, which is the usual way to run this.
    p.add_argument("--cases", nargs="*", default=None,
                   help="specific frame ids, e.g. AD_20832_85. Omit to render "
                        "every frame of --fold-split.")
    p.add_argument("--fold-split", choices=("train", "test", "all"),
                   default="test",
                   help="which of the checkpoint's subjects to render")
    p.add_argument("--subjects", nargs="*",
                   help="restrict to these subjects")
    p.add_argument("--stride", type=int, default=1,
                   help="render every Nth frame (1 = all)")
    p.add_argument("--limit", type=int, help="stop after this many frames")
    p.add_argument("--positive-name", default=None,
                   help="display name for class 1 (default: from the config)")
    p.add_argument("--negative-name", default=None)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def paint(shape, patches, coordinates, values=None):
    """Lay patches on a canvas; with ``values``, one constant per patch."""
    canvas = np.full(shape, np.nan, dtype=np.float32)
    for index, (patch, (row, column)) in enumerate(zip(patches, coordinates)):
        height, width = patch.shape[:2]
        block = patch if values is None else values[index]
        canvas[row : row + height, column : column + width] = block
    return canvas


def render(path, patches, coordinates, attention, contribution, title,
           positive, negative, background=None):
    """Three panels on one canvas: tissue, attention, signed contribution.

    Two deliberate scale choices, both because attention on this cohort is
    extremely concentrated (median max alpha 0.47 over ~253 patches):

    * attention is drawn as ``alpha / alpha.max()``, NOT as a percentile rank.
      A rank map is uniform by construction, so it makes a frame decided by one
      patch look identical to a frame with evenly spread attention. That is the
      single most misleading thing one can do to an attention map.
    * contribution uses a symmetric-log norm. The top patch can hold >95% of
      the total, so a linear scale renders every other patch as neutral grey.
      Symlog keeps the sign, keeps zero neutral, and still shows the smaller
      evidence instead of hiding it.
    """
    from matplotlib.colors import SymLogNorm

    rows = [r + p.shape[0] for p, (r, _) in zip(patches, coordinates)]
    columns = [c + p.shape[1] for p, (_, c) in zip(patches, coordinates)]
    shape = (max(rows), max(columns))
    if background is not None:
        shape = (max(shape[0], background.shape[0]),
                 max(shape[1], background.shape[1]))

    tissue = paint(shape, patches, coordinates)
    if background is not None:
        full = np.full(shape, np.nan, dtype=np.float32)
        full[: background.shape[0], : background.shape[1]] = background
        tissue = full
    relative = np.asarray(attention, dtype=float)
    relative = relative / (relative.max() or 1.0)
    attention_map = paint(shape, patches, coordinates, relative)

    # Percent of the frame's total evidence, signed:
    #     pct_i = 100 * c_i / sum_j |c_j|
    # The absolute values then sum to 100 by construction, so a patch's number
    # reads directly as "this patch is 97% of why the model decided", and the
    # scale is identical in every frame - unlike raw logits, whose range moves
    # with how confident the frame happens to be.
    signed = np.asarray(contribution, dtype=float)
    magnitude = np.abs(signed)
    total = float(magnitude.sum()) or 1.0
    percent = 100.0 * signed / total
    contribution_map = paint(shape, patches, coordinates, percent)

    limit = 100.0
    threshold = 0.1  # below 0.1% is noise; keep the colour linear there

    figure, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    axes[0].imshow(tissue, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Tissue", fontsize=11)

    axes[1].imshow(tissue, cmap="gray", vmin=0, vmax=1)
    a = axes[1].imshow(attention_map, cmap="magma", vmin=0, vmax=1, alpha=0.60)
    share = float(np.sort(relative / relative.sum())[::-1][:5].sum())
    axes[1].set_title(
        f"Attention  (magnitude only)\ntop-5 patches hold {share:.0%} of the weight",
        fontsize=11,
    )
    bar_a = figure.colorbar(a, ax=axes[1], fraction=0.046, pad=0.03)
    bar_a.set_label(r"$\alpha\ /\ \alpha_{max}$", fontsize=8)
    bar_a.ax.tick_params(labelsize=7)

    axes[2].imshow(tissue, cmap="gray", vmin=0, vmax=1)
    c = axes[2].imshow(
        contribution_map, cmap="RdBu_r", alpha=0.60,
        norm=SymLogNorm(linthresh=threshold, vmin=-limit, vmax=limit, base=10),
    )
    top = float(np.abs(percent).max())
    axes[2].set_title(
        "Contribution  (% of the frame's evidence, signed)\n"
        f"single largest patch holds {top:.0f}%",
        fontsize=11,
    )
    bar_c = figure.colorbar(
        c, ax=axes[2], fraction=0.046, pad=0.03,
        ticks=[-100, -10, -1, -0.1, 0, 0.1, 1, 10, 100],
    )
    bar_c.ax.set_yticklabels(
        ["100%", "10%", "1%", "0.1%", "0", "0.1%", "1%", "10%", "100%"]
    )
    bar_c.set_label(f"<- {negative}        0        {positive} ->", fontsize=8)
    bar_c.ax.tick_params(labelsize=7)

    for axis in axes:
        axis.axis("off")
    figure.suptitle(title, fontsize=12.5, y=1.0)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    checkpoint, model = load_checkpoint(args.checkpoint, device)
    config = checkpoint["args"]
    positive = args.positive_name or config["label1"]
    negative = args.negative_name or config["label0"]

    if args.cases:
        subjects = sorted({subject_from_stem(case) for case in args.cases})
    else:
        subjects = {
            "train": checkpoint["train_subjects"],
            "test": checkpoint["test_subjects"],
            "all": checkpoint["train_subjects"] + checkpoint["test_subjects"],
        }[args.fold_split]
    if args.subjects:
        wanted = set(args.subjects)
        subjects = [s for s in subjects if s in wanted]
    if not subjects:
        raise ValueError("No subjects selected")

    dataset = OCTBags(
        sorted(set(subjects)), args.features_dir,
        config["label0"], config["label1"],
        patch_size=config.get("patch_size", 112),
        require_coordinates=bool(config.get("spatial_mode", "none") != "none"),
    )
    by_case = {s["case_id"]: s for s in dataset.samples}

    if args.cases:
        cases = list(args.cases)
    else:
        cases = sorted(
            by_case,
            key=lambda c: (subject_from_stem(c), int(c.rsplit("_", 1)[1])),
        )
        if args.stride > 1:
            cases = cases[:: args.stride]
    print(f"{len(cases)} frames to render from {len(set(subjects))} subjects")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for case in cases:
        if args.limit is not None and saved >= args.limit:
            break
        sample = by_case.get(case)
        if sample is None:
            print(f"skipping {case}: no feature bag")
            continue
        patch_path = args.patches_dir / f"{case}.pkl"
        if not patch_path.exists():
            print(f"skipping {case}: no patches at {patch_path}")
            continue
        with patch_path.open("rb") as handle:
            payload = pickle.load(handle)

        background = None
        if args.frames_dir is not None:
            frame_path = args.frames_dir / f"{case}.mat"
            if frame_path.exists():
                import scipy.io as sio

                from .preprocess import normalize_image

                payload_mat = sio.loadmat(frame_path)
                if "frame" not in payload_mat:
                    raise KeyError(f"'frame' missing from {frame_path}")
                background = normalize_image(
                    np.asarray(payload_mat["frame"]).squeeze()
                )
            else:
                print(f"{case}: no .mat frame, falling back to the patch canvas")

        coordinates = sample.get("coordinates")
        contributions, raw_attention, logits = patch_contributions(
            model, sample["bag"].to(device),
            coordinates.to(device) if coordinates is not None else None,
        )
        signed = margin(contributions.cpu().numpy())
        attention = torch.softmax(raw_attention, dim=0).cpu().numpy()
        if len(attention) != len(payload["patches"]):
            raise ValueError(
                f"{case}: {len(attention)} features vs "
                f"{len(payload['patches'])} patches"
            )

        truth = positive if int(sample["label"]) == 1 else negative
        guess = positive if int(logits.argmax()) == 1 else negative
        total = float(logits[0, 1] - logits[0, 0])
        title = (
            f"{case}   true {truth}   predicted {guess}"
            f"{'' if truth == guess else '  (WRONG)'}   "
            f"logit margin {total:+.2f}"
        )
        render(args.output_dir / f"{case}.png", payload["patches"],
               payload["coordinates"], attention, signed, title,
               positive, negative, background=background)
        saved += 1
        if saved % 25 == 0 or saved == len(cases):
            print(f"  {saved}/{len(cases)}", flush=True)
    print(f"Saved {saved} map figures to {args.output_dir}")


if __name__ == "__main__":
    main()
