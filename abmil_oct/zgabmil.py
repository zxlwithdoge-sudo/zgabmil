"""GABMIL with a z-axis added to the Spatial Information Mixing Module.

GABMIL's SIMM mixes patch *features* along two in-plane axes: BLOCK (a local
window) and GRID (a dilated, global sampling). Both live inside a single frame,
which is the right structure for a whole-slide image, where a slide is the only
spatial context there is.

An OCT volume has a third axis. On this cohort it is the strongest one: the same
(row, column) in consecutive frames has feature cosine 0.988 versus 0.953 for
in-plane neighbours, i.e. about 3x the excess similarity over shuffled pairs.
The in-plane ablation hurt the baseline with a clean dose-response, while an
inference-only smoothing along z helped (+0.022 frame AUC, 9/10 folds), so the
axis GABMIL does not use is the one carrying the signal.

This module adds Z as a third mixing stage, built exactly like the other two: a
channel-shared linear map over positions, followed by a residual. It mixes
features, not attention.

    dense -> [BLOCK] -> [GRID] -> [Z] -> attention -> pool -> classify

What is deliberately NOT changed, so results stay comparable to the manuscript
baseline and the GABMIL ablation table:

* the bag is still one frame - neighbouring frames enter only as context,
* the attention softmax is still over one frame's patches,
* the loss is still per frame and the subject prediction still a majority vote.

Only the features of the target frame are altered, by information from its
neighbours.

Two details worth knowing before reading results:

* Coordinates must come from ``data.absolute_grid_coordinates``. The default
  per-frame convention re-origins every frame separately, which misaligns 4-13%
  of adjacent frame pairs on this cohort. ``OCTContextBags`` handles this.
* Frames missing at the edges of a volume are zero-padded, the same treatment
  the reference gives to empty lattice cells. The z window is therefore always
  ``2 * z_window + 1`` wide and the linear map has a fixed shape. Whether
  masking those positions helps is untested.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .model import (
    GatedAttention,
    PositionMixer,
    SPATIAL_MODES,
    grid_partition,
    grid_reverse,
    window_partition,
    window_reverse,
)


class ZSpatialMixer(nn.Module):
    """SIMM over a stack of consecutive frames, with an added z stage.

    Parameters
    ----------
    mode : str
        In-plane mixing, one of ``SPATIAL_MODES``. ``"none"`` runs the z stage
        alone, which is the arm the evidence on this cohort points at.
    z_window : int
        Half-width along the frame axis; the mixer sees ``2 * z_window + 1``
        frames. 0 disables the z stage, which makes this module equivalent to
        the original per-frame ``SpatialMixer``.
    """

    def __init__(self, mode="none", block_size=3, grid_size=3, z_window=1,
                 dropout=0.25):
        super().__init__()
        if mode not in SPATIAL_MODES:
            raise ValueError(f"mode must be one of {SPATIAL_MODES}, got '{mode}'")
        self.mode = mode
        self.use_block = mode in ("block", "block_grid")
        self.use_grid = mode in ("grid", "block_grid")
        self.z_window = int(z_window)
        if self.z_window < 0:
            raise ValueError(f"z_window must be >= 0, got {z_window}")
        self.use_z = self.z_window > 0
        if not (self.use_block or self.use_grid or self.use_z):
            raise ValueError(
                "ZSpatialMixer does nothing: set an in-plane mode or z_window > 0"
            )

        self.block_size = int(block_size) if self.use_block else 1
        self.grid_size = int(grid_size) if self.use_grid else 1
        if self.block_size < 1 or self.grid_size < 1:
            raise ValueError("block_size and grid_size must be >= 1")
        self.pad_multiple = math.lcm(self.block_size, self.grid_size)

        if self.use_block:
            self.block_mlp = PositionMixer(
                self.block_size * self.block_size, dropout=dropout
            )
        if self.use_grid:
            self.grid_mlp = PositionMixer(
                self.grid_size * self.grid_size, dropout=dropout
            )
        if self.use_z:
            # Same construction as BLOCK and GRID, over the frame axis instead.
            self.z_mlp = PositionMixer(self.z_positions, dropout=dropout)
        self.reset_parameters()

    @property
    def z_positions(self):
        return 2 * self.z_window + 1

    def reset_parameters(self):
        """Xavier-normal weights and zero bias, as in the GABMIL reference."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def extra_repr(self):
        return (
            f"mode={self.mode}, block_size={self.block_size}, "
            f"grid_size={self.grid_size}, z_window={self.z_window}"
        )

    def _scatter(self, instances, coordinates, frames, n_frames):
        """Place every context frame's patches on ONE shared lattice.

        The origin is taken over the whole stack, never per frame: a per-frame
        origin shifts frames relative to each other and destroys the very
        correspondence this module exists to use.
        """
        rows = coordinates[:, 0] - coordinates[:, 0].min()
        columns = coordinates[:, 1] - coordinates[:, 1].min()
        height = int(rows.max().item()) + 1
        width = int(columns.max().item()) + 1
        padded_height = self.pad_multiple * math.ceil(height / self.pad_multiple)
        padded_width = self.pad_multiple * math.ceil(width / self.pad_multiple)

        dense = torch.zeros(
            (n_frames, padded_height, padded_width, instances.shape[1]),
            dtype=instances.dtype,
            device=instances.device,
        )
        dense[frames, rows, columns] = instances
        return dense, rows, columns

    def _mix_plane(self, dense, partition, reverse, mixer, size):
        """Apply an in-plane branch to each frame of the stack."""
        outputs = []
        for index in range(dense.shape[0]):
            frame = dense[index]
            height, width, channels = frame.shape
            groups = partition(frame, size)
            groups = groups.view(-1, size * size, channels).permute(0, 2, 1)
            groups = mixer(groups)
            groups = groups.permute(0, 2, 1).reshape(-1, size, size, channels)
            outputs.append(reverse(groups, (height, width), size))
        return torch.stack(outputs, dim=0)

    def _mix_z(self, dense):
        """Mix along the frame axis: [F, H, W, C] -> [F, H, W, C]."""
        n_frames, height, width, channels = dense.shape
        # [H*W, C, F] so the linear map runs over frame positions, shared across
        # channels and lattice cells - the same contract PositionMixer has for
        # BLOCK and GRID.
        positions = dense.permute(1, 2, 3, 0).reshape(-1, channels, n_frames)
        positions = self.z_mlp(positions)
        return positions.reshape(height, width, channels, n_frames).permute(
            3, 0, 1, 2
        )

    def forward(self, instances, coordinates, frames, n_frames):
        """Mix ``[N, C]`` patches drawn from ``n_frames`` consecutive frames.

        ``frames`` holds each patch's frame index within the stack (0-based).
        Returns the mixed features in the input order.
        """
        if instances.ndim != 2:
            raise ValueError(
                f"Expected [instances, features], got {tuple(instances.shape)}"
            )
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError(
                f"Expected [instances, 2] coordinates, got {tuple(coordinates.shape)}"
            )
        if coordinates.shape[0] != instances.shape[0]:
            raise ValueError(
                "Coordinate/feature count mismatch: "
                f"{coordinates.shape[0]} vs {instances.shape[0]}"
            )
        if self.use_z and n_frames != self.z_positions:
            raise ValueError(
                f"z_window={self.z_window} needs exactly {self.z_positions} "
                f"context frames (zero-padded at volume edges), got {n_frames}"
            )

        coordinates = coordinates.to(device=instances.device, dtype=torch.long)
        frames = frames.to(device=instances.device, dtype=torch.long)
        dense, rows, columns = self._scatter(
            instances, coordinates, frames, n_frames
        )

        # --- in-plane stages, exactly as in the reference ---
        if self.use_block:
            blocked = self._mix_plane(
                dense, window_partition, window_reverse,
                self.block_mlp, self.block_size,
            )
        else:
            blocked = dense

        if self.use_grid:
            if self.use_block:
                blocked = blocked + dense
            gridded = self._mix_plane(
                blocked, grid_partition, grid_reverse,
                self.grid_mlp, self.grid_size,
            )
            mixed = blocked + gridded
        elif self.use_block:
            mixed = dense + blocked
        else:
            mixed = dense

        # --- z stage, same shape of construction, residual as above ---
        if self.use_z:
            mixed = mixed + self._mix_z(mixed)

        return mixed[frames, rows, columns]


class GABMILZ(nn.Module):
    """ABMIL whose features are mixed in-plane and along z before attention.

    The bag is still a single frame. ``forward`` receives that frame's patches
    plus its neighbours, mixes them together, and then attends and classifies
    using only the target frame's patches, so every downstream quantity -
    per-frame softmax, per-frame loss, subject majority vote - is unchanged from
    the baseline.
    """

    def __init__(
        self,
        feature_dim=1024,
        embedding_dim=512,
        attention_dim=256,
        dropout=0.25,
        num_classes=2,
        spatial_mode="none",
        block_size=3,
        grid_size=3,
        z_window=1,
    ):
        super().__init__()
        self.spatial_mode = spatial_mode
        self.z_window = int(z_window)
        self.spatial_mixer = ZSpatialMixer(
            mode=spatial_mode, block_size=block_size, grid_size=grid_size,
            z_window=z_window, dropout=dropout,
        )

        embedding = [nn.Linear(feature_dim, embedding_dim), nn.ReLU()]
        if dropout:
            embedding.append(nn.Dropout(dropout))
        self.embedding = nn.Sequential(*embedding)
        self.attention = GatedAttention(
            input_dim=embedding_dim, hidden_dim=attention_dim, dropout=dropout
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    @property
    def uses_coordinates(self):
        return True

    @property
    def context_frames(self):
        return 2 * self.z_window + 1

    def forward(self, bag, coordinates, frames, target_frame):
        """``bag`` [N, feature_dim] over all context frames.

        ``frames`` [N] is each patch's index in the stack, ``target_frame`` the
        index of the frame being classified. Returns the same tuple as
        ``OCTAttention.forward``.
        """
        bag, coordinates, frames = _squeeze_batch(bag, coordinates, frames)
        n_frames = self.context_frames if self.z_window > 0 else 1

        mixed = self.spatial_mixer(bag, coordinates, frames, n_frames)

        selection = frames == int(target_frame)
        if not bool(selection.any()):
            raise ValueError(
                f"target frame {int(target_frame)} holds no patches; the "
                "dataset must always place the target at a populated index"
            )
        target = mixed[selection]

        embedded = self.embedding(target)
        raw_attention = self.attention(embedded).transpose(1, 0)
        attention = F.softmax(raw_attention, dim=1)
        pooled = attention @ embedded
        logits = self.classifier(pooled)
        return logits, F.softmax(logits, dim=1), logits.argmax(dim=1), raw_attention


def _squeeze_batch(bag, coordinates, frames):
    """Accept DataLoader batch_size=1 tensors as well as bare ones."""
    if bag.ndim == 3:
        if bag.shape[0] != 1:
            raise ValueError("GABMILZ requires DataLoader batch_size=1")
        bag = bag.squeeze(0)
    if coordinates.ndim == 3:
        coordinates = coordinates.squeeze(0)
    if frames.ndim == 2:
        frames = frames.squeeze(0)
    return bag, coordinates, frames


def build_zgabmil(config, device=None):
    """Instantiate the model described by a training ``run_config`` mapping."""
    model = GABMILZ(
        feature_dim=config.get("feature_dim", 1024),
        embedding_dim=config.get("embedding_dim", 512),
        attention_dim=config.get("attention_dim", 256),
        dropout=config.get("dropout", 0.25),
        num_classes=config.get("num_classes", 2),
        spatial_mode=config.get("spatial_mode", "none"),
        block_size=config.get("block_size", 3),
        grid_size=config.get("grid_size", 3),
        z_window=config.get("z_window", 1),
    )
    return model if device is None else model.to(device)
