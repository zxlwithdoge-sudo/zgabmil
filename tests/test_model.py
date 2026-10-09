import pytest
import torch

from abmil_oct.model import OCTAttention


def test_model_shapes_and_attention_normalization():
    model = OCTAttention(dropout=0.0)
    bag = torch.randn(1, 7, 1024)
    logits, probabilities, prediction, raw_attention = model(bag)

    assert logits.shape == (1, 2)
    assert probabilities.shape == (1, 2)
    assert prediction.shape == (1,)
    assert raw_attention.shape == (1, 7)
    assert torch.allclose(
        probabilities.sum(dim=1), torch.ones(1), atol=1e-6
    )
    assert torch.allclose(
        torch.softmax(raw_attention, dim=1).sum(dim=1),
        torch.ones(1),
        atol=1e-6,
    )


def test_default_model_has_no_spatial_parameters():
    """spatial_mode='none' must leave the manuscript ABMIL model untouched."""
    torch.manual_seed(0)
    baseline = OCTAttention(dropout=0.0)
    torch.manual_seed(0)
    explicit = OCTAttention(dropout=0.0, spatial_mode="none")

    assert baseline.spatial_mixer is None
    assert not baseline.uses_coordinates
    assert [name for name, _ in baseline.named_parameters()] == [
        name for name, _ in explicit.named_parameters()
    ]

    bag = torch.randn(11, 1024)
    baseline.eval()
    explicit.eval()
    with torch.no_grad():
        assert torch.equal(baseline(bag)[0], explicit(bag)[0])


@pytest.mark.parametrize("mode", ["block", "grid", "block_grid"])
def test_spatial_model_shapes(mode):
    model = OCTAttention(dropout=0.0, spatial_mode=mode, block_size=3, grid_size=2)
    model.eval()
    rows, columns = torch.meshgrid(
        torch.arange(5), torch.arange(4), indexing="ij"
    )
    coordinates = torch.stack([rows.reshape(-1), columns.reshape(-1)], dim=1)
    bag = torch.randn(coordinates.shape[0], 1024)

    logits, probabilities, prediction, raw_attention = model(bag, coordinates)

    assert model.uses_coordinates
    assert logits.shape == (1, 2)
    assert probabilities.shape == (1, 2)
    assert prediction.shape == (1,)
    assert raw_attention.shape == (1, coordinates.shape[0])


def test_spatial_model_requires_coordinates():
    model = OCTAttention(dropout=0.0, spatial_mode="block")
    with pytest.raises(ValueError, match="coordinates"):
        model(torch.randn(6, 1024))


def test_batched_inputs_are_squeezed():
    model = OCTAttention(dropout=0.0, spatial_mode="block", block_size=2)
    model.eval()
    coordinates = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    bag = torch.randn(4, 1024)
    with torch.no_grad():
        flat = model(bag, coordinates)[0]
        batched = model(bag.unsqueeze(0), coordinates.unsqueeze(0))[0]
    assert torch.allclose(flat, batched, atol=1e-6)


def test_invalid_spatial_mode():
    with pytest.raises(ValueError, match="spatial_mode"):
        OCTAttention(spatial_mode="diagonal")
