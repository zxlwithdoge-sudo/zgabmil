"""Frame bags served together with their neighbouring frames.

``OCTBags`` hands the model one frame at a time, which is all plain ABMIL and
in-plane GABMIL need. ``zgabmil.GABMILZ`` mixes features along the frame axis,
so it needs a frame's neighbours as well. This module groups a subject's frames
back into the volume they came from and serves each frame with the +/- w frames
around it.

The bag definition does not change: one frame, one label, one loss term. Only
the context travels with it.

Coordinates come from ``data.absolute_grid_coordinates``, never the per-frame
convention: re-origining each frame separately shifts frames relative to each
other, which on this cohort misaligns 4-13% of adjacent frame pairs by a patch
and silently destroys the correspondence the z stage exists to use.
"""

import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import absolute_grid_coordinates, subject_from_stem

FRAME_SUFFIX = re.compile(r"_(\d+)$")


def frame_index(case_id):
    """``AD_20832_137`` -> 137. Frames are consecutive slices through a volume."""
    match = FRAME_SUFFIX.search(case_id)
    if not match:
        raise ValueError(
            f"Cannot read a frame number from '{case_id}'; z mixing needs frames "
            "named <subject>_<index>"
        )
    return int(match.group(1))


def load_volumes(subjects, features_dir, label0, label1, patch_size=112):
    """Read every requested subject's frames, keyed by frame number.

    Returns ``{subject: {"order": [...], "frames": {k: {...}}, "label": int,
    "gaps": int}}``. ``gaps`` counts breaks in the frame numbering: two frames
    that are numerically adjacent but not physically adjacent, which the z stage
    would otherwise treat as neighbours.
    """
    features_dir = Path(features_dir)
    wanted = set(subjects)

    def label_of(subject):
        if subject.startswith(f"{label0}_"):
            return 0
        if subject.startswith(f"{label1}_"):
            return 1
        raise ValueError(f"Subject {subject} does not match configured labels")

    by_subject = defaultdict(dict)
    for path in sorted(features_dir.glob("*.pkl")):
        subject = subject_from_stem(path.stem)
        if subject not in wanted:
            continue
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        features = np.asarray(payload["features"], dtype=np.float32)
        if features.ndim != 2 or features.shape[0] == 0:
            continue
        raw = payload.get("coordinates")
        if raw is None:
            raise KeyError(
                f"'coordinates' missing from {path}; z mixing needs patch "
                "positions. Re-run abmil_oct.features."
            )
        coordinates = absolute_grid_coordinates(raw, patch_size=patch_size)
        if coordinates.shape[0] != features.shape[0]:
            raise ValueError(f"Coordinate/feature count mismatch in {path}")
        by_subject[subject][frame_index(path.stem)] = {
            "features": torch.from_numpy(features),
            "coordinates": coordinates,
            "case_id": path.stem,
        }

    missing = wanted - set(by_subject)
    if missing:
        raise FileNotFoundError(
            f"No valid feature bags for subjects: {sorted(missing)}"
        )

    volumes = {}
    for subject, frames in by_subject.items():
        order = sorted(frames)
        volumes[subject] = {
            "order": order,
            "frames": frames,
            "label": label_of(subject),
            "gaps": sum(1 for a, b in zip(order, order[1:]) if b - a != 1),
        }
    return volumes


class OCTContextBags(Dataset):
    """One bag per frame, served with its +/- ``z_window`` neighbouring frames.

    The target frame always sits at stack index ``z_window``. Neighbours outside
    the volume simply contribute no patches, leaving their slice of the lattice
    at zero - the treatment the GABMIL reference gives to empty lattice cells.
    """

    def __init__(
        self,
        subjects,
        features_dir,
        label0,
        label1,
        patch_size=112,
        z_window=1,
    ):
        if z_window < 0:
            raise ValueError(f"z_window must be >= 0, got {z_window}")
        self.z_window = int(z_window)
        self.volumes = load_volumes(
            subjects, features_dir, label0, label1, patch_size=patch_size
        )
        self.entries = [
            (subject, key)
            for subject in sorted(self.volumes)
            for key in self.volumes[subject]["order"]
        ]
        if not self.entries:
            raise ValueError("No frames matched the requested subjects")
        # train.make_sampler reads this.
        self.samples = [
            {"label": self.volumes[subject]["label"]} for subject, _ in self.entries
        ]

    @property
    def subjects_with_gaps(self):
        return sorted(s for s, v in self.volumes.items() if v["gaps"])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        subject, key = self.entries[index]
        volume = self.volumes[subject]
        features, coordinates, frames = [], [], []

        for offset in range(-self.z_window, self.z_window + 1):
            neighbour = volume["frames"].get(key + offset)
            if neighbour is None:
                continue  # outside the volume: that slice stays zero
            position = offset + self.z_window
            features.append(neighbour["features"])
            coordinates.append(neighbour["coordinates"])
            frames.append(
                torch.full(
                    (neighbour["features"].shape[0],), position, dtype=torch.long
                )
            )

        return {
            "bag": torch.cat(features, dim=0),
            "coordinates": torch.cat(coordinates, dim=0),
            "frames": torch.cat(frames, dim=0),
            "target_frame": self.z_window,
            "label": torch.tensor(volume["label"], dtype=torch.long),
            "case_id": volume["frames"][key]["case_id"],
            "subject_id": subject,
        }


def collate_single(batch):
    """Identity collate for batch_size=1; bags have ragged patch counts."""
    if len(batch) != 1:
        raise ValueError("z mixing requires batch_size=1")
    return batch[0]
