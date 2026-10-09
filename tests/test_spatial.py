import numpy as np
import pytest
import torch

from abmil_oct.data import grid_coordinates, infer_patch_stride
from abmil_oct.model import (
    SpatialMixer,
    grid_partition,
    grid_reverse,
    window_partition,
    window_reverse,
)


def test_window_partition_roundtrip():
    features = torch.randn(6, 9, 5)
    windows = window_partition(features, 3)
    assert windows.shape == (6, 3, 3, 5)
    assert torch.equal(window_reverse(windows, (6, 9), 3), features)


def test_grid_partition_roundtrip():
    features = torch.randn(6, 9, 5)
    grid = grid_partition(features, 3)
    assert grid.shape == (6, 3, 3, 5)
    assert torch.equal(grid_reverse(grid, (6, 9), 3), features)


def test_window_and_grid_sample_different_positions():
    """BLOCK groups neighbours; GRID groups positions spread across the frame."""
    features = torch.arange(36, dtype=torch.float32).view(6, 6, 1)
    window = window_partition(features, 3)[0, :, :, 0]
    grid = grid_partition(features, 3)[0, :, :, 0]

    assert window.tolist() == [[0, 1, 2], [6, 7, 8], [12, 13, 14]]
    assert grid.tolist() == [[0, 2, 4], [12, 14, 16], [24, 26, 28]]


def test_mixer_preserves_patch_order_and_count():
    mixer = SpatialMixer(mode="block_grid", block_size=2, grid_size=3, dropout=0.0)
    mixer.eval()
    coordinates = torch.tensor([[0, 0], [2, 3], [1, 1], [4, 0]])
    instances = torch.randn(4, 8)
    with torch.no_grad():
        mixed = mixer(instances, coordinates)
    assert mixed.shape == instances.shape

    # Permuting the bag permutes the output identically.
    order = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        permuted = mixer(instances[order], coordinates[order])
    assert torch.allclose(permuted, mixed[order], atol=1e-6)


def test_mixer_is_translation_invariant():
    """Absolute pixel offsets must not change the result, only the layout."""
    mixer = SpatialMixer(mode="block", block_size=3, dropout=0.0)
    mixer.eval()
    coordinates = torch.tensor([[0, 0], [0, 1], [1, 0], [2, 2]])
    instances = torch.randn(4, 6)
    with torch.no_grad():
        near = mixer(instances, coordinates)
        far = mixer(instances, coordinates + torch.tensor([7, 11]))
    assert torch.allclose(near, far, atol=1e-6)


def test_mixer_uses_neighbouring_patches():
    """A change in one patch must reach its block neighbours."""
    torch.manual_seed(0)
    mixer = SpatialMixer(mode="block", block_size=3, dropout=0.0)
    mixer.eval()
    coordinates = torch.tensor([[0, 0], [0, 1], [0, 2]])
    instances = torch.zeros(3, 4)
    with torch.no_grad():
        baseline = mixer(instances, coordinates)
        instances[0] = 1.0
        perturbed = mixer(instances, coordinates)
    assert not torch.allclose(baseline[1], perturbed[1], atol=1e-6)


@pytest.mark.parametrize("mode", ["block", "grid", "block_grid"])
def test_mixer_gradients_flow(mode):
    mixer = SpatialMixer(mode=mode, block_size=2, grid_size=2, dropout=0.0)
    coordinates = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1], [3, 2]])
    instances = torch.randn(5, 4, requires_grad=True)
    mixer(instances, coordinates).sum().backward()
    assert instances.grad is not None
    for parameter in mixer.parameters():
        assert parameter.grad is not None


def test_mixer_rejects_bad_shapes():
    mixer = SpatialMixer(mode="block", block_size=2, dropout=0.0)
    with pytest.raises(ValueError, match="instances, 2"):
        mixer(torch.randn(3, 4), torch.zeros(3, 3, dtype=torch.long))
    with pytest.raises(ValueError, match="mismatch"):
        mixer(torch.randn(3, 4), torch.zeros(2, 2, dtype=torch.long))


def test_infer_patch_stride_and_grid_coordinates():
    pixels = [(0, 672), (112, 560), (224, 112), (2464, 1568)]
    assert infer_patch_stride(np.asarray(pixels)) == 112

    grid = grid_coordinates(pixels)
    assert grid.dtype == torch.long
    assert grid.tolist() == [[0, 5], [1, 4], [2, 0], [22, 13]]

    explicit = grid_coordinates(pixels, patch_size=112)
    assert torch.equal(grid, explicit)


def test_grid_coordinates_single_patch():
    grid = grid_coordinates([(336, 448)])
    assert grid.tolist() == [[0, 0]]
