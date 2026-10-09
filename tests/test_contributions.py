"""The contribution decomposition must be exact for BOTH heads.

If ``sum_i c_i != logits`` the map is an approximation dressed up as an
attribution, which is the thing this module exists to avoid.
"""

import numpy as np
import pytest
import torch

from abmil_oct.additive import AdditiveOCTAttention
from abmil_oct.contributions import (
    divergence,
    margin,
    patch_contributions,
)
from abmil_oct.model import OCTAttention


def bag(n=40, dim=8, seed=0):
    torch.manual_seed(seed)
    return torch.randn(n, dim)


def test_linear_head_decomposes_exactly_including_the_bias():
    """OCTAttention already supports exact attribution - no retraining."""
    torch.manual_seed(0)
    model = OCTAttention(feature_dim=8, embedding_dim=16, attention_dim=8,
                         dropout=0.0).eval()
    with torch.no_grad():                      # a non-zero bias is the trap
        model.classifier.bias.copy_(torch.tensor([0.7, -1.3]))
    x = bag(dim=8)
    contributions, _, logits = patch_contributions(model, x)
    reference = model(x)[0]
    assert torch.allclose(logits, reference, atol=1e-5)
    assert torch.allclose(contributions.sum(0, keepdim=True), reference,
                          atol=1e-5)


def test_additive_head_decomposes_exactly():
    torch.manual_seed(0)
    model = AdditiveOCTAttention(feature_dim=8, embedding_dim=16,
                                 attention_dim=8, dropout=0.0,
                                 head_hidden=(16,)).eval()
    x = bag(dim=8)
    contributions, _, logits = patch_contributions(model, x)
    assert torch.allclose(logits, model(x)[0], atol=1e-5)
    assert torch.allclose(contributions.sum(0, keepdim=True), logits, atol=1e-5)


def test_spatial_mixing_path_decomposes_exactly():
    torch.manual_seed(0)
    model = OCTAttention(feature_dim=8, embedding_dim=16, attention_dim=8,
                         dropout=0.0, spatial_mode="block", block_size=2).eval()
    x = bag(n=12, dim=8)
    coordinates = torch.tensor([(r, c) for r in range(4) for c in range(3)])
    contributions, _, logits = patch_contributions(model, x, coordinates)
    assert torch.allclose(contributions.sum(0, keepdim=True),
                          model(x, coordinates)[0], atol=1e-5)


def test_spatial_checkpoint_without_coordinates_is_rejected():
    model = OCTAttention(feature_dim=8, embedding_dim=16, attention_dim=8,
                         dropout=0.0, spatial_mode="block", block_size=2).eval()
    with pytest.raises(ValueError, match="coordinates"):
        patch_contributions(model, bag(n=12, dim=8))


def test_margin_is_signed_and_matches_the_prediction():
    torch.manual_seed(0)
    model = OCTAttention(feature_dim=8, embedding_dim=16, attention_dim=8,
                         dropout=0.0).eval()
    x = bag(n=100, dim=8)
    contributions, _, logits = patch_contributions(model, x)
    m = margin(contributions.cpu().numpy())
    assert m.shape == (100,)
    # the margin sums to the logit margin, so its sign agrees with argmax
    total = m.sum()
    assert (total > 0) == (int(logits.argmax()) == 1)


def test_divergence_reports_opposing_attended_patches():
    rng = np.random.default_rng(0)
    attention = rng.normal(size=200)
    contribution = rng.normal(size=200)      # independent of attention
    stats = divergence(attention, contribution, predicted_class=1)
    assert stats["n_patches"] == 200
    assert 0.0 <= stats["opposing_fraction"] <= 1.0
    # independent inputs -> roughly half the attended patches oppose
    assert 0.3 < stats["opposing_fraction"] < 0.7
    assert abs(stats["spearman_attention_vs_signed"]) < 0.3


def test_divergence_is_zero_when_the_maps_agree():
    attention = np.linspace(-1, 1, 100)
    stats = divergence(attention, attention, predicted_class=1)
    assert stats["opposing_fraction"] == 0.0
    assert stats["spearman_attention_vs_signed"] == pytest.approx(1.0)


def test_too_few_patches_returns_none():
    assert divergence(np.array([0.1, 0.2]), np.array([1.0, -1.0]), 1) is None
