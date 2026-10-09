"""Gated-attention MIL for OCT frames, with optional GABMIL spatial mixing.

``OCTAttention`` is the manuscript ABMIL model. Setting ``spatial_mode`` to
something other than ``"none"`` inserts the Spatial Information Mixing Module
(SIMM) of GABMIL (Keshvarikhojasteh et al., arXiv:2504.17379,
https://github.com/hassancpu/GABMIL) between the frozen patch features and the
attention branch. The module places
the patch features back on the frame's patch grid and mixes positions with two
lightweight MLPs:

``block``
    Local mixing inside non-overlapping ``block_size x block_size`` windows.
``grid``
    Global mixing over a dilated ``grid_size x grid_size`` sampling grid.
``block_grid``
    Both, applied in sequence (BLOCK then GRID) as in the reference code.

The mixer is purely additive: with ``spatial_mode="none"`` no extra parameters
are created and the forward pass is bit-for-bit identical to the published
ABMIL-OCT model, so the manuscript results remain reproducible.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


SPATIAL_MODES = ("none", "block", "grid", "block_grid")


def window_partition(features, window_size):
    """Split ``[H, W, C]`` into ``[windows, window_size, window_size, C]``."""
    height, width, channels = features.shape
    windows = features.view(
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    windows = windows.permute(0, 2, 1, 3, 4).contiguous()
    return windows.view(-1, window_size, window_size, channels)


def window_reverse(windows, original_size, window_size):
    """Inverse of :func:`window_partition`."""
    height, width = original_size
    output = windows.view(
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        -1,
    )
    output = output.permute(0, 2, 1, 3, 4).contiguous()
    return output.view(height, width, -1)


def grid_partition(features, grid_size):
    """Split ``[H, W, C]`` into ``[grids, grid_size, grid_size, C]``.

    Unlike :func:`window_partition` the sampled positions are spread across the
    whole frame, which is what gives the GRID branch its global receptive
    field.
    """
    height, width, channels = features.shape
    grid = features.view(
        grid_size,
        height // grid_size,
        grid_size,
        width // grid_size,
        channels,
    )
    grid = grid.permute(1, 3, 0, 2, 4).contiguous()
    return grid.view(-1, grid_size, grid_size, channels)


def grid_reverse(grid, original_size, grid_size):
    """Inverse of :func:`grid_partition`."""
    height, width = original_size
    channels = grid.shape[-1]
    output = grid.view(
        height // grid_size,
        width // grid_size,
        grid_size,
        grid_size,
        channels,
    )
    output = output.permute(2, 0, 3, 1, 4).contiguous()
    return output.view(height, width, channels)


class PositionMixer(nn.Module):
    """Channel-wise MLP that mixes positions inside a window or grid."""

    def __init__(self, num_positions, expansion_factor=1.0, dropout=0.25):
        super().__init__()
        hidden_positions = max(1, int(num_positions * expansion_factor))
        self.fc = nn.Linear(num_positions, hidden_positions)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, positions):
        # positions: [groups, channels, num_positions]
        return F.relu(self.dropout(self.fc(positions)))


class SpatialMixer(nn.Module):
    """GABMIL Spatial Information Mixing Module for a single OCT frame.

    Parameters
    ----------
    mode:
        One of ``SPATIAL_MODES``.
    block_size:
        Side length, in patches, of the local BLOCK window.
    grid_size:
        Side length, in patches, of the dilated GRID window.
    dropout:
        Dropout applied inside each mixing MLP.
    """

    def __init__(self, mode="block", block_size=3, grid_size=3, dropout=0.25):
        super().__init__()
        if mode not in SPATIAL_MODES:
            raise ValueError(
                f"spatial_mode must be one of {SPATIAL_MODES}, got '{mode}'"
            )
        if mode == "none":
            raise ValueError("SpatialMixer requires a mode other than 'none'")

        self.mode = mode
        self.use_block = mode in ("block", "block_grid")
        self.use_grid = mode in ("grid", "block_grid")
        self.block_size = int(block_size) if self.use_block else 1
        self.grid_size = int(grid_size) if self.use_grid else 1
        if self.block_size < 1 or self.grid_size < 1:
            raise ValueError("block_size and grid_size must be >= 1")
        # The grid must be padded to a size both partitions can divide.
        self.pad_multiple = math.lcm(self.block_size, self.grid_size)

        if self.use_block:
            self.block_mlp = PositionMixer(
                self.block_size * self.block_size, dropout=dropout
            )
        if self.use_grid:
            self.grid_mlp = PositionMixer(
                self.grid_size * self.grid_size, dropout=dropout
            )
        self.reset_parameters()

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
            f"grid_size={self.grid_size}"
        )

    def _scatter(self, instances, coordinates):
        """Place ``[N, C]`` patch features on a zero-padded patch grid."""
        rows = coordinates[:, 0] - coordinates[:, 0].min()
        columns = coordinates[:, 1] - coordinates[:, 1].min()
        height = int(rows.max().item()) + 1
        width = int(columns.max().item()) + 1
        padded_height = self.pad_multiple * math.ceil(height / self.pad_multiple)
        padded_width = self.pad_multiple * math.ceil(width / self.pad_multiple)

        dense = torch.zeros(
            (padded_height, padded_width, instances.shape[1]),
            dtype=instances.dtype,
            device=instances.device,
        )
        dense[rows, columns] = instances
        return dense, rows, columns

    def _mix(self, dense, partition, reverse, mixer, size):
        height, width, channels = dense.shape
        groups = partition(dense, size)
        groups = groups.view(-1, size * size, channels).permute(0, 2, 1)
        groups = mixer(groups)
        groups = groups.permute(0, 2, 1).reshape(-1, size, size, channels)
        return reverse(groups, (height, width), size)

    def forward(self, instances, coordinates):
        """Mix ``[N, C]`` patch features and return them in the input order."""
        if instances.ndim != 2:
            raise ValueError(
                f"Expected [instances, features], got {tuple(instances.shape)}"
            )
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError(
                "Expected [instances, 2] grid coordinates, got "
                f"{tuple(coordinates.shape)}"
            )
        if coordinates.shape[0] != instances.shape[0]:
            raise ValueError(
                "Coordinate/feature count mismatch: "
                f"{coordinates.shape[0]} vs {instances.shape[0]}"
            )

        coordinates = coordinates.to(device=instances.device, dtype=torch.long)
        dense, rows, columns = self._scatter(instances, coordinates)

        if self.use_block:
            blocked = self._mix(
                dense,
                window_partition,
                window_reverse,
                self.block_mlp,
                self.block_size,
            )
        else:
            blocked = dense

        if self.use_grid:
            if self.use_block:
                blocked = blocked + dense
            gridded = self._mix(
                blocked,
                grid_partition,
                grid_reverse,
                self.grid_mlp,
                self.grid_size,
            )
            mixed = blocked + gridded
        else:
            mixed = dense + blocked

        # Drop the padding and restore the original patch ordering so that the
        # attention weights stay aligned with the patch bag.
        return mixed[rows, columns]


class GatedAttention(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, dropout=0.25):
        super().__init__()
        branch_a = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        branch_b = [nn.Linear(input_dim, hidden_dim), nn.Sigmoid()]
        if dropout:
            branch_a.append(nn.Dropout(dropout))
            branch_b.append(nn.Dropout(dropout))
        self.branch_a = nn.Sequential(*branch_a)
        self.branch_b = nn.Sequential(*branch_b)
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, instances):
        return self.score(self.branch_a(instances) * self.branch_b(instances))


class OCTAttention(nn.Module):
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
    ):
        super().__init__()
        if spatial_mode not in SPATIAL_MODES:
            raise ValueError(
                f"spatial_mode must be one of {SPATIAL_MODES}, "
                f"got '{spatial_mode}'"
            )
        self.spatial_mode = spatial_mode
        self.spatial_mixer = (
            None
            if spatial_mode == "none"
            else SpatialMixer(
                mode=spatial_mode,
                block_size=block_size,
                grid_size=grid_size,
                dropout=dropout,
            )
        )

        embedding = [nn.Linear(feature_dim, embedding_dim), nn.ReLU()]
        if dropout:
            embedding.append(nn.Dropout(dropout))
        self.embedding = nn.Sequential(*embedding)
        self.attention = GatedAttention(
            input_dim=embedding_dim,
            hidden_dim=attention_dim,
            dropout=dropout,
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    @property
    def uses_coordinates(self):
        return self.spatial_mixer is not None

    def forward(self, bag, coordinates=None):
        if bag.ndim == 3:
            if bag.shape[0] != 1:
                raise ValueError("ABMIL requires DataLoader batch_size=1")
            bag = bag.squeeze(0)
        if bag.ndim != 2:
            raise ValueError(f"Expected [instances, features], got {bag.shape}")

        if self.spatial_mixer is not None:
            if coordinates is None:
                raise ValueError(
                    f"spatial_mode='{self.spatial_mode}' requires patch grid "
                    "coordinates; pass coordinates=<[instances, 2] tensor>"
                )
            if coordinates.ndim == 3:
                if coordinates.shape[0] != 1:
                    raise ValueError("GABMIL requires DataLoader batch_size=1")
                coordinates = coordinates.squeeze(0)
            bag = self.spatial_mixer(bag, coordinates)

        embedded = self.embedding(bag)
        raw_attention = self.attention(embedded).transpose(1, 0)
        attention = F.softmax(raw_attention, dim=1)
        pooled = attention @ embedded
        logits = self.classifier(pooled)
        probabilities = F.softmax(logits, dim=1)
        prediction = logits.argmax(dim=1)
        return logits, probabilities, prediction, raw_attention


def build_model(config, device=None):
    """Instantiate the model described by a training ``run_config`` mapping."""
    model = OCTAttention(
        feature_dim=config.get("feature_dim", 1024),
        embedding_dim=config.get("embedding_dim", 512),
        attention_dim=config.get("attention_dim", 256),
        dropout=config.get("dropout", 0.25),
        num_classes=config.get("num_classes", 2),
        spatial_mode=config.get("spatial_mode", "none"),
        block_size=config.get("block_size", 3),
        grid_size=config.get("grid_size", 3),
    )
    return model if device is None else model.to(device)
