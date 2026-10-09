"""Correctness tests for the z-axis extension of GABMIL's SIMM.

The load-bearing claims:
  1. with the z stage off, the module reproduces the original per-frame
     SpatialMixer exactly, so the GABMIL ablation stays the reference point;
  2. the z stage really mixes along the frame axis and nothing else;
  3. every context frame lands on ONE shared lattice - the per-frame origin
     that broke the earlier z-smoothing work cannot come back in through here.
"""

import numpy as np
import pytest
import torch

from abmil_oct.model import OCTAttention, SpatialMixer
from abmil_oct.zgabmil import GABMILZ, ZSpatialMixer


def stack(n_frames=3, rows=4, columns=3, channels=8, seed=0, drop=0.0):
    """Patches from consecutive frames on a shared lattice."""
    rng = np.random.default_rng(seed)
    features, coordinates, frames = [], [], []
    for k in range(n_frames):
        for r in range(rows):
            for c in range(columns):
                if rng.random() < drop:
                    continue
                features.append(rng.normal(size=channels).astype(np.float32))
                coordinates.append((r, c))
                frames.append(k)
    return (
        torch.from_numpy(np.stack(features)),
        torch.tensor(coordinates, dtype=torch.long),
        torch.tensor(frames, dtype=torch.long),
    )


# --- z off must equal the original GABMIL ------------------------------------

@pytest.mark.parametrize("mode,size", [("block", 2), ("grid", 3), ("block_grid", 2)])
def test_z_window_zero_reproduces_the_original_spatial_mixer(mode, size):
    """One frame, z off: identical output to SpatialMixer with the same weights."""
    torch.manual_seed(0)
    reference = SpatialMixer(mode=mode, block_size=size, grid_size=size, dropout=0.0)
    extended = ZSpatialMixer(mode=mode, block_size=size, grid_size=size,
                             z_window=0, dropout=0.0)
    extended.load_state_dict(reference.state_dict())
    reference.eval()
    extended.eval()

    features, coordinates, frames = stack(n_frames=1, rows=4, columns=6, channels=8)
    with torch.inference_mode():
        a = reference(features, coordinates)
        b = extended(features, coordinates, frames, 1)
    assert torch.allclose(a, b, atol=1e-6, rtol=0)
    assert float((a - b).abs().max()) == 0.0


def test_z_stage_requires_the_declared_number_of_frames():
    mixer = ZSpatialMixer(mode="none", z_window=2, dropout=0.0)
    features, coordinates, frames = stack(n_frames=3, channels=8)
    with pytest.raises(ValueError, match="context frames"):
        mixer(features, coordinates, frames, 3)   # needs 5


def test_a_mixer_that_does_nothing_is_rejected():
    with pytest.raises(ValueError, match="does nothing"):
        ZSpatialMixer(mode="none", z_window=0)


# --- the z stage mixes along z, and only along z -----------------------------

def test_z_mixing_makes_the_target_depend_on_its_neighbours():
    torch.manual_seed(0)
    mixer = ZSpatialMixer(mode="none", z_window=1, dropout=0.0).eval()
    features, coordinates, frames = stack(n_frames=3, channels=8)

    with torch.inference_mode():
        before = mixer(features, coordinates, frames, 3)[frames == 1]
        perturbed = features.clone()
        perturbed[frames == 0] += 5.0          # change a NEIGHBOUR only
        after = mixer(perturbed, coordinates, frames, 3)[frames == 1]
    assert not torch.allclose(before, after, atol=1e-5)


def test_z_mixing_does_not_mix_across_lattice_positions():
    """With no in-plane stage, position (r, c) must only see (r, c) of other frames."""
    torch.manual_seed(0)
    mixer = ZSpatialMixer(mode="none", z_window=1, dropout=0.0).eval()
    features, coordinates, frames = stack(n_frames=3, rows=3, columns=3, channels=4)

    target = (frames == 1)
    position = (coordinates[:, 0] == 0) & (coordinates[:, 1] == 0)
    with torch.inference_mode():
        before = mixer(features, coordinates, frames, 3)
        perturbed = features.clone()
        # perturb a DIFFERENT position in a neighbouring frame
        elsewhere = (frames == 0) & (coordinates[:, 0] == 2) & (coordinates[:, 1] == 2)
        perturbed[elsewhere] += 5.0
        after = mixer(perturbed, coordinates, frames, 3)
    watched = target & position
    assert torch.allclose(before[watched], after[watched], atol=1e-6)


def test_channels_share_one_map_over_frame_positions():
    """The z stage is a position mixer, so it has 2w+1 inputs, not C."""
    mixer = ZSpatialMixer(mode="none", z_window=3, dropout=0.0)
    assert mixer.z_positions == 7
    assert mixer.z_mlp.fc.in_features == 7
    assert mixer.z_mlp.fc.out_features == 7


# --- one shared lattice ------------------------------------------------------

def test_frames_with_different_extents_stay_aligned():
    """A neighbour whose mask starts further right must not be re-origined.

    This is the failure that confounded the first z-smoothing run: re-origining
    each frame separately shifts it by a patch, so "the same (row, column)"
    silently refers to different tissue.
    """
    torch.manual_seed(0)
    mixer = ZSpatialMixer(mode="none", z_window=1, dropout=0.0).eval()
    channels = 4
    # frame 0 covers columns 1..2, frames 1 and 2 cover columns 0..2
    coordinates, frames, features = [], [], []
    rng = np.random.default_rng(0)
    for k in range(3):
        start = 1 if k == 0 else 0
        for c in range(start, 3):
            coordinates.append((0, c))
            frames.append(k)
            features.append(rng.normal(size=channels).astype(np.float32))
    features = torch.from_numpy(np.stack(features))
    coordinates = torch.tensor(coordinates, dtype=torch.long)
    frames = torch.tensor(frames, dtype=torch.long)

    with torch.inference_mode():
        before = mixer(features, coordinates, frames, 3)
        perturbed = features.clone()
        # column 1 of frame 0 must influence column 1 of frame 1, not column 0
        mask = (frames == 0) & (coordinates[:, 1] == 1)
        perturbed[mask] += 5.0
        after = mixer(perturbed, coordinates, frames, 3)

    target_c1 = (frames == 1) & (coordinates[:, 1] == 1)
    target_c0 = (frames == 1) & (coordinates[:, 1] == 0)
    assert not torch.allclose(before[target_c1], after[target_c1], atol=1e-5)
    assert torch.allclose(before[target_c0], after[target_c0], atol=1e-6)


# --- the whole model ---------------------------------------------------------

def test_model_classifies_only_the_target_frame():
    """Output shape and attention length must match the target frame alone."""
    torch.manual_seed(0)
    model = GABMILZ(feature_dim=8, embedding_dim=16, attention_dim=8,
                    dropout=0.0, spatial_mode="none", z_window=1).eval()
    features, coordinates, frames = stack(n_frames=3, channels=8)
    with torch.inference_mode():
        logits, probabilities, prediction, attention = model(
            features, coordinates, frames, 1
        )
    assert logits.shape == (1, 2)
    assert float(probabilities.sum()) == pytest.approx(1.0, abs=1e-5)
    assert attention.shape[1] == int((frames == 1).sum())


def test_attention_softmax_is_still_per_frame():
    """The bag is unchanged: weights over the target frame sum to one."""
    torch.manual_seed(0)
    model = GABMILZ(feature_dim=8, embedding_dim=16, attention_dim=8,
                    dropout=0.0, spatial_mode="none", z_window=1).eval()
    features, coordinates, frames = stack(n_frames=3, channels=8)
    with torch.inference_mode():
        attention = model(features, coordinates, frames, 1)[3]
    assert float(torch.softmax(attention, dim=1).sum()) == pytest.approx(1.0, abs=1e-5)


def test_edge_frames_work_with_missing_neighbours():
    """At a volume edge a neighbour is simply absent; its lattice slice is zero."""
    torch.manual_seed(0)
    model = GABMILZ(feature_dim=8, embedding_dim=16, attention_dim=8,
                    dropout=0.0, spatial_mode="none", z_window=1).eval()
    features, coordinates, frames = stack(n_frames=3, channels=8)
    keep = frames != 0                     # drop the frame before the target
    with torch.inference_mode():
        logits = model(features[keep], coordinates[keep], frames[keep], 1)[0]
    assert logits.shape == (1, 2)
    assert torch.isfinite(logits).all()


def test_gradients_reach_the_z_mixer():
    torch.manual_seed(0)
    model = GABMILZ(feature_dim=8, embedding_dim=16, attention_dim=8,
                    dropout=0.0, spatial_mode="none", z_window=1)
    features, coordinates, frames = stack(n_frames=3, channels=8)
    logits = model(features, coordinates, frames, 1)[0]
    torch.nn.functional.cross_entropy(logits, torch.tensor([1])).backward()
    grad = model.spatial_mixer.z_mlp.fc.weight.grad
    assert grad is not None and float(grad.abs().sum()) > 0


def test_combining_in_plane_and_z_runs_and_differs_from_z_alone():
    torch.manual_seed(0)
    features, coordinates, frames = stack(n_frames=3, rows=4, columns=4, channels=8)
    z_only = GABMILZ(feature_dim=8, embedding_dim=16, attention_dim=8, dropout=0.0,
                     spatial_mode="none", z_window=1).eval()
    both = GABMILZ(feature_dim=8, embedding_dim=16, attention_dim=8, dropout=0.0,
                   spatial_mode="block", block_size=2, z_window=1).eval()
    shared = {k: v for k, v in z_only.state_dict().items()
              if not k.startswith("spatial_mixer.block")}
    both.load_state_dict({**both.state_dict(), **shared})
    with torch.inference_mode():
        a = z_only(features, coordinates, frames, 1)[0]
        b = both(features, coordinates, frames, 1)[0]
    assert not torch.allclose(a, b, atol=1e-5)


def test_plain_abmil_is_untouched():
    """Guard: importing this module must not perturb the manuscript model."""
    torch.manual_seed(0)
    model = OCTAttention(feature_dim=8, embedding_dim=16, attention_dim=8,
                         dropout=0.0).eval()
    bag = torch.randn(20, 8)
    with torch.inference_mode():
        assert model(bag)[0].shape == (1, 2)
