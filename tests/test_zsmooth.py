"""Correctness tests for the z-smoothed attention evaluation.

The load-bearing claim of zsmooth_eval is that w=0 is *exactly* the stock model.
The script rebuilds the forward pass by hand (embedding -> attention -> softmax
-> pool -> classifier) instead of calling model(bag), so that claim is a property
of the code, not of any particular checkpoint, and it is tested here.
"""

import numpy as np
import pytest
import torch

from abmil_oct.model import OCTAttention
from abmil_oct.zsmooth_eval import frame_index, smooth_logits


def manual_forward(model, bag):
    """The decomposition zsmooth_eval.encode_fold + evaluate_window perform."""
    embedded = model.embedding(bag)
    logits = model.attention(embedded).reshape(-1)
    attention = torch.softmax(logits, dim=0)
    pooled = attention.unsqueeze(0) @ embedded
    return model.classifier(pooled)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_manual_forward_matches_model_forward(seed):
    """w=0 must reproduce the stock model bit-for-bit, for any weights."""
    torch.manual_seed(seed)
    model = OCTAttention(dropout=0.25).eval()
    bag = torch.randn(137, 1024)

    with torch.inference_mode():
        reference = model(bag)[0]
        rebuilt = manual_forward(model, bag)

    assert torch.allclose(reference, rebuilt, atol=1e-6, rtol=0)
    assert int(reference.argmax(1)) == int(rebuilt.argmax(1))


def test_manual_forward_matches_after_training_shifts_attention():
    """Same equivalence once attention is far from uniform (a trained model)."""
    torch.manual_seed(0)
    model = OCTAttention(dropout=0.0)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
    bag = torch.randn(80, 1024)
    target = torch.tensor([1])
    for _ in range(60):  # drive attention away from uniform
        loss = torch.nn.functional.cross_entropy(model(bag)[0], target)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    model.eval()

    with torch.inference_mode():
        weights = torch.softmax(model.attention(model.embedding(bag)).reshape(-1), 0)
        assert weights.max() / weights.min() > 1.5, "attention still ~uniform; test is vacuous"
        assert torch.allclose(model(bag)[0], manual_forward(model, bag), atol=1e-6)


def make_frames(n_frames=5, rows=4, cols=3, seed=0, drop=0.0):
    """Synthetic subject: consecutive frames on a shared (row, col) lattice."""
    rng = np.random.default_rng(seed)
    frames = {}
    for k in range(n_frames):
        coords, logits = [], []
        for r in range(rows):
            for c in range(cols):
                if rng.random() < drop:
                    continue
                coords.append((r, c))
                logits.append(float(rng.normal()))
        frames[k] = {
            "coords": np.array(coords, dtype=np.int64),
            "logit": torch.tensor(logits, dtype=torch.float32),
        }
    return frames


def test_window_zero_is_identity():
    frames = make_frames()
    for k in frames:
        assert smooth_logits(frames, k, 0) is frames[k]["logit"]


def test_smoothing_is_the_mean_of_available_neighbours():
    """Hand-check one patch against an explicitly computed average."""
    frames = make_frames(n_frames=5, drop=0.0)
    k, w = 2, 1
    smoothed = smooth_logits(frames, k, w)
    position = tuple(frames[k]["coords"][0])

    expected = []
    for offset in (-1, 0, 1):
        neighbour = frames[k + offset]
        index = [i for i, rc in enumerate(neighbour["coords"])
                 if tuple(rc) == position][0]
        expected.append(float(neighbour["logit"][index]))
    assert smoothed[0].item() == pytest.approx(float(np.mean(expected)), abs=1e-6)


def test_edge_frames_use_fewer_neighbours():
    """Frame 0 has no k-1, so it averages over 2 frames, not 3."""
    frames = make_frames(n_frames=5, drop=0.0)
    position = tuple(frames[0]["coords"][0])
    expected = []
    for k in (0, 1):
        index = [i for i, rc in enumerate(frames[k]["coords"])
                 if tuple(rc) == position][0]
        expected.append(float(frames[k]["logit"][index]))
    assert smooth_logits(frames, 0, 1)[0].item() == pytest.approx(np.mean(expected), abs=1e-6)


def test_missing_positions_are_skipped_not_zero_filled():
    """A patch absent from the neighbour must not be averaged against 0.

    Zero-filling would systematically pull tissue-edge patches toward zero
    attention, which is an artefact, not signal.
    """
    frames = make_frames(n_frames=3, drop=0.0)
    position = tuple(frames[1]["coords"][0])
    # delete that position from both neighbours
    for k in (0, 2):
        keep = [i for i, rc in enumerate(frames[k]["coords"])
                if tuple(rc) != position]
        frames[k]["coords"] = frames[k]["coords"][keep]
        frames[k]["logit"] = frames[k]["logit"][keep]

    smoothed = smooth_logits(frames, 1, 1)
    assert smoothed[0].item() == pytest.approx(float(frames[1]["logit"][0]), abs=1e-6)


def test_gaps_in_frame_numbering_reduce_neighbour_count():
    """Non-consecutive frame indices must be detected, not silently averaged."""
    frames = make_frames(n_frames=5, drop=0.0)
    del frames[3]
    stats = {"neighbours": [], "attention_shift": [], }
    smooth_logits(frames, 2, 1, stats)
    smooth_logits(frames, 4, 1, stats)
    assert stats["neighbours"] == [1, 0]  # frame 2 sees only k=1; frame 4 sees nothing


def test_isolated_frame_is_unchanged():
    frames = make_frames(n_frames=1)
    stats = {"neighbours": [], "attention_shift": [], }
    out = smooth_logits(frames, 0, 2, stats)
    assert torch.equal(out, frames[0]["logit"])
    assert stats["attention_shift"] == [0.0]


def test_frame_index_parsing():
    assert frame_index("AD_20832_137") == 137
    assert frame_index("NC_6839_0") == 0
    assert frame_index("CTE_8572_scan_5") == 5
