import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio
from scipy import ndimage
from skimage.filters import threshold_otsu

from .provenance import sha256_file, software_versions, write_json


def normalize_image(image):
    """Robustly rescale positive finite pixels to [0, 1] per frame."""
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros_like(image)

    image = np.where(finite, image, 0.0)
    positive = image[image > 0]
    if positive.size == 0:
        return np.zeros_like(image)

    low, high = np.percentile(positive, [1, 99])
    if high <= low:
        low, high = float(positive.min()), float(positive.max())
    if high <= low:
        return np.zeros_like(image)

    normalized = np.clip((image - low) / (high - low), 0.0, 1.0)
    normalized[image <= 0] = 0.0
    return normalized.astype(np.float32)


def compute_tissue_mask(
    image,
    median_size=15,
    otsu_scale=0.35,
    min_component_size=1024,
):
    """Compute the tissue mask used by the manuscript pipeline."""
    smoothed = ndimage.median_filter(image, size=median_size)
    positive = smoothed[smoothed > 0]
    if positive.size == 0:
        return np.zeros_like(image, dtype=bool)

    threshold = max(0.02, otsu_scale * threshold_otsu(positive))
    mask = smoothed > threshold
    structure = np.ones((3, 3), dtype=bool)
    mask = ndimage.binary_opening(mask, structure=structure)
    mask = ndimage.binary_closing(mask, structure=structure)

    labels, count = ndimage.label(mask)
    if count == 0:
        return mask

    sizes = np.bincount(labels.ravel())
    keep = np.flatnonzero(sizes >= min_component_size)
    keep = keep[keep != 0]
    if keep.size == 0:
        return np.zeros_like(mask)
    return ndimage.binary_fill_holes(np.isin(labels, keep))


def extract_patches(
    image,
    patch_size=112,
    median_size=15,
    otsu_scale=0.35,
    min_tissue_fraction=0.2,
    min_component_size=1024,
):
    normalized = normalize_image(image)
    tissue_mask = compute_tissue_mask(
        normalized,
        median_size=median_size,
        otsu_scale=otsu_scale,
        min_component_size=min_component_size,
    )

    patches, coordinates, tissue_fractions = [], [], []
    height, width = normalized.shape
    for row in range(0, height - patch_size + 1, patch_size):
        for column in range(0, width - patch_size + 1, patch_size):
            patch_mask = tissue_mask[
                row : row + patch_size,
                column : column + patch_size,
            ]
            fraction = float(patch_mask.mean())
            if fraction < min_tissue_fraction or not patch_mask.any():
                continue

            patch = normalized[
                row : row + patch_size,
                column : column + patch_size,
            ]
            patch = cv2.resize(
                patch,
                (patch_size, patch_size),
                interpolation=cv2.INTER_LINEAR,
            )
            patches.append(patch.astype(np.float32))
            coordinates.append((row, column))
            tissue_fractions.append(fraction)

    if not patches:
        return None
    return {
        "patches": np.stack(patches),
        "coordinates": coordinates,
        "tissue_fractions": tissue_fractions,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Normalize OCT frames and create tissue-filtered patches"
    )
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--patches-dir", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, default=112)
    parser.add_argument("--median-size", type=int, default=15)
    parser.add_argument("--otsu-scale", type=float, default=0.35)
    parser.add_argument("--min-tissue-fraction", type=float, default=0.2)
    parser.add_argument("--min-component-size", type=int, default=1024)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.patches_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = sorted(args.frames_dir.glob("*.mat"))
    if not frame_paths:
        raise FileNotFoundError(f"No .mat files found in {args.frames_dir}")

    parameters = {
        "patch_size": args.patch_size,
        "median_size": args.median_size,
        "otsu_scale": args.otsu_scale,
        "min_tissue_fraction": args.min_tissue_fraction,
        "min_component_size": args.min_component_size,
    }
    expected_outputs = {f"{path.stem}.pkl" for path in frame_paths}
    orphaned_outputs = {
        path.name for path in args.patches_dir.glob("*.pkl")
    } - expected_outputs
    if orphaned_outputs:
        raise RuntimeError(
            "Patch directory contains outputs without matching MAT frames: "
            f"{sorted(orphaned_outputs)}. Use a fresh output directory."
        )

    saved, empty = 0, 0
    frame_records = []
    for frame_path in frame_paths:
        output_path = args.patches_dir / f"{frame_path.stem}.pkl"
        source_sha256 = sha256_file(frame_path)
        if args.skip_existing and output_path.exists():
            with output_path.open("rb") as handle:
                existing = pickle.load(handle)
            provenance = existing.get("provenance", {})
            if (
                provenance.get("source_sha256") != source_sha256
                or provenance.get("parameters") != parameters
            ):
                raise RuntimeError(
                    f"Existing patches do not match the current source or "
                    f"parameters: {output_path}"
                )
            frame_shape = next(
                (
                    list(shape)
                    for name, shape, _ in sio.whosmat(frame_path)
                    if name == "frame"
                ),
                None,
            )
            frame_records.append(
                {
                    "file": frame_path.name,
                    "sha256": source_sha256,
                    "shape": frame_shape,
                    "patches": len(existing["patches"]),
                    "output": output_path.name,
                    "output_sha256": sha256_file(output_path),
                }
            )
            saved += 1
            continue

        contents = sio.loadmat(frame_path)
        if "frame" not in contents:
            raise KeyError(f"'frame' array missing from {frame_path}")
        image = np.asarray(contents["frame"]).squeeze()
        if image.ndim != 2:
            raise ValueError(
                f"Expected a 2D frame in {frame_path}, got {image.shape}"
            )

        patch_data = extract_patches(
            image,
            patch_size=args.patch_size,
            median_size=args.median_size,
            otsu_scale=args.otsu_scale,
            min_tissue_fraction=args.min_tissue_fraction,
            min_component_size=args.min_component_size,
        )
        if patch_data is None:
            if output_path.exists():
                raise RuntimeError(
                    f"{frame_path.name} now produces no patches, but "
                    f"{output_path} already exists. Use a fresh output "
                    "directory."
                )
            empty += 1
            frame_records.append(
                {
                    "file": frame_path.name,
                    "sha256": source_sha256,
                    "shape": list(image.shape),
                    "patches": 0,
                }
            )
            print(f"No qualifying tissue patches: {frame_path.name}")
            continue

        patch_data["provenance"] = {
            "source_file": frame_path.name,
            "source_sha256": source_sha256,
            "parameters": parameters,
        }
        with output_path.open("wb") as handle:
            pickle.dump(patch_data, handle)
        frame_records.append(
            {
                "file": frame_path.name,
                "sha256": source_sha256,
                "shape": list(image.shape),
                "patches": len(patch_data["patches"]),
                "output": output_path.name,
                "output_sha256": sha256_file(output_path),
            }
        )
        saved += 1
        print(f"{frame_path.name}: saved {len(patch_data['patches'])} patches")

    write_json(
        args.patches_dir / "preprocessing_manifest.json",
        {
            "schema_version": 1,
            "source_directory": str(args.frames_dir.resolve()),
            "parameters": parameters,
            "software": software_versions(),
            "frames": frame_records,
            "saved_frame_bags": saved,
            "frames_without_patches": empty,
        },
    )
    print(f"Finished: {saved} frame bags available, {empty} frames without patches")


if __name__ == "__main__":
    main()
