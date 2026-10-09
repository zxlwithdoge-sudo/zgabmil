import pickle
import re
from math import gcd
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


SUBJECT_PATTERN = re.compile(r"^([A-Za-z]+_\d+)(?:_|$)")


def subject_from_stem(stem):
    match = SUBJECT_PATTERN.match(stem)
    if not match:
        raise ValueError(
            f"Cannot parse subject from '{stem}'; expected LABEL_NUMBER_*"
        )
    return match.group(1)


def discover_subjects(features_dir, labels=None):
    subjects = {
        subject_from_stem(path.stem)
        for path in Path(features_dir).glob("*.pkl")
    }
    if labels:
        prefixes = tuple(f"{label}_" for label in labels)
        subjects = {subject for subject in subjects if subject.startswith(prefixes)}
    return sorted(subjects)


def infer_patch_stride(coordinates):
    """Recover the patch stride in pixels from a bag's pixel coordinates."""
    offsets = np.asarray(coordinates, dtype=np.int64)
    offsets = offsets - offsets.min(axis=0, keepdims=True)
    stride = 0
    for value in np.unique(offsets):
        stride = gcd(stride, int(value))
    return stride


def grid_coordinates(coordinates, patch_size=None):
    """Convert pixel ``(row, column)`` patch origins to integer grid indices.

    The preprocessing step emits non-overlapping patch origins, so dividing by
    the patch stride yields the frame's patch lattice. ``patch_size`` may be
    given explicitly; otherwise it is inferred from the coordinates themselves,
    which also handles frames whose patches all sit in one row or column.
    """
    pixels = np.asarray(coordinates, dtype=np.int64)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError(
            f"Expected [instances, 2] coordinates, got {pixels.shape}"
        )
    stride = int(patch_size) if patch_size else infer_patch_stride(pixels)
    if stride <= 0:
        # A single patch, or every patch at the same origin.
        stride = 1
    origin = pixels.min(axis=0, keepdims=True)
    return torch.as_tensor((pixels - origin) // stride, dtype=torch.long)


def absolute_grid_coordinates(coordinates, patch_size=None):
    """Patch-grid indices on a lattice shared by every frame of a subject.

    ``grid_coordinates`` subtracts each frame's *own* bounding box origin, which
    is correct for models that only ever look inside one frame (GABMIL), but
    wrong the moment two frames are compared: a frame whose tissue mask starts
    one patch further right gets its whole lattice shifted by one column, so
    "the same (row, column)" silently refers to different tissue.

    On this cohort the per-frame origin is not constant within a subject, and
    4-13% of adjacent frame pairs disagree (measured on AD_20832, AD_10382,
    NC_21499), so any cross-frame model must use this function instead.

    Dividing the raw pixel origin by the stride keeps every frame on one global
    lattice. Patch origins here are exact multiples of the patch size, so the
    division is lossless.
    """
    pixels = np.asarray(coordinates, dtype=np.int64)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError(
            f"Expected [instances, 2] coordinates, got {pixels.shape}"
        )
    if (pixels < 0).any():
        raise ValueError("Pixel coordinates must be non-negative")
    stride = int(patch_size) if patch_size else infer_patch_stride(pixels)
    if stride <= 0:
        stride = 1
    return torch.as_tensor(pixels // stride, dtype=torch.long)


GRID_ORIGINS = ("frame", "absolute")


class OCTBags(Dataset):
    """Frame-level bags of patch features, optionally with grid coordinates.

    ``patch_size`` is only used to translate pixel coordinates into patch-grid
    indices for the GABMIL spatial mixer; it does not affect plain ABMIL.
    """

    def __init__(
        self,
        subjects,
        features_dir,
        label0,
        label1,
        patch_size=None,
        require_coordinates=False,
        grid_origin="frame",
    ):
        if grid_origin not in GRID_ORIGINS:
            raise ValueError(
                f"grid_origin must be one of {GRID_ORIGINS}, got '{grid_origin}'"
            )
        self.features_dir = Path(features_dir)
        self.label0 = label0
        self.label1 = label1
        self.patch_size = patch_size
        self.require_coordinates = require_coordinates
        self.grid_origin = grid_origin
        self.samples = self._load_samples(set(subjects))
        if not self.samples:
            raise ValueError("No feature bags matched the requested subjects")

        dimensions = {sample["bag"].shape[1] for sample in self.samples}
        if len(dimensions) != 1:
            raise ValueError(f"Inconsistent feature dimensions: {dimensions}")
        self.feature_dim = dimensions.pop()

    def _label(self, subject):
        if subject.startswith(f"{self.label0}_"):
            return 0
        if subject.startswith(f"{self.label1}_"):
            return 1
        raise ValueError(f"Subject {subject} does not match configured labels")

    def _load_samples(self, subjects):
        samples = []
        found_subjects = set()
        for path in sorted(self.features_dir.glob("*.pkl")):
            subject = subject_from_stem(path.stem)
            if subject not in subjects:
                continue
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            if "features" not in payload:
                raise KeyError(f"'features' missing from {path}")
            bag = torch.as_tensor(payload["features"], dtype=torch.float32)
            if bag.ndim != 2 or bag.shape[0] == 0:
                continue

            raw_coordinates = payload.get("coordinates")
            if raw_coordinates is None:
                if self.require_coordinates:
                    raise KeyError(
                        f"'coordinates' missing from {path}; spatial models "
                        "need patch positions. Re-run abmil_oct.features."
                    )
                coordinates = None
            else:
                convert = (
                    grid_coordinates
                    if self.grid_origin == "frame"
                    else absolute_grid_coordinates
                )
                coordinates = convert(
                    raw_coordinates, patch_size=self.patch_size
                )
                if coordinates.shape[0] != bag.shape[0]:
                    raise ValueError(
                        f"Coordinate/feature count mismatch in {path}"
                    )

            samples.append(
                {
                    "bag": bag,
                    "coordinates": coordinates,
                    "label": self._label(subject),
                    "case_id": path.stem,
                    "subject_id": subject,
                }
            )
            found_subjects.add(subject)

        missing = subjects - found_subjects
        if missing:
            raise FileNotFoundError(
                f"No valid feature bags for subjects: {sorted(missing)}"
            )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        coordinates = sample["coordinates"]
        if coordinates is None:
            # Keep the default collate happy for coordinate-free bags.
            coordinates = torch.zeros((sample["bag"].shape[0], 2), dtype=torch.long)
        return {
            "bag": sample["bag"],
            "coordinates": coordinates,
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "case_id": sample["case_id"],
            "subject_id": sample["subject_id"],
        }
