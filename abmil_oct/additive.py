"""Additive MIL head: every patch scores itself, and the scores are summed.

Javed et al., *Additive MIL: Intrinsically Interpretable Multiple Instance
Learning for Pathology*, NeurIPS 2022, move the sum outside the predictor::

    classical  g(x) = p( sum_i  alpha_i f(x_i) )          (their Eq. 3)
    additive   g(x) = sum_i  p( alpha_i f(x_i) )          (their Eq. 5)

so the bag logits decompose exactly into per-patch, per-class, signed
contributions. Their Theorem 1 states those contributions are proportional to
the patch's Shapley value.

Why this file exists even though ``OCTAttention`` is already additive
---------------------------------------------------------------------
``OCTAttention.classifier`` is a single ``nn.Linear``, and a linear map
commutes with the sum, so the manuscript model *already* admits the exact
decomposition - we measured max |classical - additive| = 1.5e-07 on it. What it
does not have is a *non-linear* per-patch scorer. This module supplies one, so
``psi`` can rank patches on something richer than a linear projection. The
interpretability is not the point here (we had that for free); the ranking
quality is, because the patch-selection work keeps only the top |contribution|.

The scaling trap
----------------
Taken literally, ``p(alpha_i * h_i)`` is a bad parameterisation. Attention is a
softmax over the bag, so with ~300 patches per OCT frame alpha_i ~ 1/300 and
||alpha_i h_i|| ~ 0.08 against ||h_i|| ~ 23. Every patch therefore lands in a
tiny neighbourhood of the origin, and a biased MLP returns almost exactly
psi(0) for all of them. Measured on random data at that scale::

    mean |c_i - psi(0)| = 0.00056      |psi(0)| = 0.0434     -> 1.3% signal
    sum_i c_i = [9.190, -16.859]   vs   N*psi(0) = [9.177, -16.853]

The contribution map would be 98.7% constant and the ranking pure noise.

The fix is to drop the biases. ReLU is positively homogeneous, so a bias-free
ReLU MLP satisfies ``psi(alpha_i h_i) == alpha_i * psi(h_i)`` exactly (verified
to 5.8e-09). We therefore compute ``alpha_i * psi(h_i)``, which is the same
function, numerically well behaved, and easier to read: psi(h_i) is the patch's
own vote and alpha_i is the weight that vote carries.

``bias=True`` is still available for ablation, but it changes the model class
rather than just the arithmetic - see ``AdditivePatchHead``.

Nothing here touches ``model.OCTAttention``; the manuscript baseline and the
GABMIL ablation stay bit-for-bit reproducible.
"""

import torch
import torch.nn.functional as F
from torch import nn

from .model import GatedAttention, SPATIAL_MODES, SpatialMixer


class AdditivePatchHead(nn.Module):
    """Per-patch class scorer ``psi``, applied before pooling.

    Parameters
    ----------
    input_dim : int
        Embedding width.
    num_classes : int
        Output width; one signed score per class per patch.
    hidden_dims : sequence of int
        Widths of the hidden layers. ``()`` gives a single linear map, which
        makes the whole model equivalent to ``OCTAttention`` (see
        ``tests/test_additive.py``) and is the correctness reference.
    dropout : float
        Dropout between hidden layers. Note this breaks exact homogeneity
        while training, which is fine - the decomposition identity is a
        property of the forward pass, and ``eval()`` restores it.
    bias : bool
        Leave this ``False``. See the module docstring: with attention-scaled
        inputs a biased head collapses to a constant. Kept only so the failure
        can be reproduced on demand.
    """

    def __init__(self, input_dim=512, num_classes=2, hidden_dims=(256,),
                 dropout=0.25, bias=False):
        super().__init__()
        self.bias = bool(bias)
        widths = [int(input_dim), *(int(w) for w in hidden_dims)]
        layers = []
        for a, b in zip(widths, widths[1:]):
            layers.append(nn.Linear(a, b, bias=self.bias))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(widths[-1], int(num_classes), bias=self.bias))
        self.psi = nn.Sequential(*layers)
        self.depth = len(hidden_dims)

    def extra_repr(self):
        return f"depth={self.depth}, bias={self.bias}"

    def forward(self, embedded, attention):
        """Signed per-class contributions ``[N, num_classes]``.

        ``attention`` is the softmax-normalised weight vector ``[1, N]``. We
        return ``alpha_i * psi(h_i)`` rather than ``psi(alpha_i * h_i)``; for a
        bias-free ReLU stack these are the same function, and for a biased one
        this is the one that still carries signal.
        """
        if embedded.ndim != 2:
            raise ValueError(
                f"Expected [instances, features], got {tuple(embedded.shape)}"
            )
        weights = attention.reshape(-1, 1)
        if weights.shape[0] != embedded.shape[0]:
            raise ValueError(
                "Attention/embedding count mismatch: "
                f"{weights.shape[0]} vs {embedded.shape[0]}"
            )
        return weights * self.psi(embedded)


class AdditiveOCTAttention(nn.Module):
    """``OCTAttention`` with the predictor moved inside the sum.

    ``forward`` returns the usual four-tuple so the training loop, evaluation
    and heatmap code need no changes. ``contributions`` exposes the per-patch
    class scores for the attribution and patch-selection experiments.
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
        head_hidden=(256,),
        head_bias=False,
    ):
        super().__init__()
        if spatial_mode not in SPATIAL_MODES:
            raise ValueError(
                f"spatial_mode must be one of {SPATIAL_MODES}, got '{spatial_mode}'"
            )
        self.spatial_mode = spatial_mode
        self.spatial_mixer = (
            None
            if spatial_mode == "none"
            else SpatialMixer(
                mode=spatial_mode, block_size=block_size,
                grid_size=grid_size, dropout=dropout,
            )
        )

        embedding = [nn.Linear(feature_dim, embedding_dim), nn.ReLU()]
        if dropout:
            embedding.append(nn.Dropout(dropout))
        self.embedding = nn.Sequential(*embedding)
        self.attention = GatedAttention(
            input_dim=embedding_dim, hidden_dim=attention_dim, dropout=dropout
        )
        self.classifier = AdditivePatchHead(
            input_dim=embedding_dim, num_classes=num_classes,
            hidden_dims=head_hidden, dropout=dropout, bias=head_bias,
        )

    @property
    def uses_coordinates(self):
        return self.spatial_mixer is not None

    def _encode(self, bag, coordinates):
        if bag.ndim == 3:
            if bag.shape[0] != 1:
                raise ValueError("Additive MIL requires DataLoader batch_size=1")
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
                coordinates = coordinates.squeeze(0)
            bag = self.spatial_mixer(bag, coordinates)

        embedded = self.embedding(bag)
        raw_attention = self.attention(embedded).transpose(1, 0)
        return embedded, raw_attention

    def contributions(self, bag, coordinates=None):
        """``[N, num_classes]`` signed contributions; they sum to the logits."""
        embedded, raw_attention = self._encode(bag, coordinates)
        attention = F.softmax(raw_attention, dim=1)
        return self.classifier(embedded, attention)

    def forward(self, bag, coordinates=None):
        embedded, raw_attention = self._encode(bag, coordinates)
        attention = F.softmax(raw_attention, dim=1)
        patch_scores = self.classifier(embedded, attention)
        logits = patch_scores.sum(dim=0, keepdim=True)
        return (
            logits,
            F.softmax(logits, dim=1),
            logits.argmax(dim=1),
            raw_attention,
        )


def build_additive_model(config, device=None):
    """Instantiate the model described by a training ``run_config`` mapping."""
    hidden = config.get("head_hidden", (256,))
    if isinstance(hidden, int):
        hidden = (hidden,)
    model = AdditiveOCTAttention(
        feature_dim=config.get("feature_dim", 1024),
        embedding_dim=config.get("embedding_dim", 512),
        attention_dim=config.get("attention_dim", 256),
        dropout=config.get("dropout", 0.25),
        num_classes=config.get("num_classes", 2),
        spatial_mode=config.get("spatial_mode", "none"),
        block_size=config.get("block_size", 3),
        grid_size=config.get("grid_size", 3),
        head_hidden=tuple(hidden),
        head_bias=config.get("head_bias", False),
    )
    return model if device is None else model.to(device)
