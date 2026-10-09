"""Correctness tests for the Additive MIL head.

The load-bearing claims:
  1. the contributions sum EXACTLY to the logits - that identity is the whole
     reason to prefer this over a post-hoc attribution;
  2. a depth-0, bias-free head reproduces ``OCTAttention`` bit-for-bit, so the
     manuscript baseline remains the reference point;
  3. a bias-free ReLU head is positively homogeneous, which is what lets us
     compute ``alpha * psi(h)`` instead of the ill-scaled ``psi(alpha * h)``;
  4. a biased head really does collapse toward a constant at realistic bag
     sizes - the failure is documented, not assumed;
  5. with hidden layers the model is genuinely NOT the classical one.
"""

import pytest
import torch

from abmil_oct.additive import AdditiveOCTAttention, AdditivePatchHead
from abmil_oct.model import OCTAttention


def bag(n=40, dim=8, seed=0):
    torch.manual_seed(seed)
    return torch.randn(n, dim)


# --- the decomposition identity ---------------------------------------------

@pytest.mark.parametrize("hidden", [(), (16,), (16, 8)])
def test_contributions_sum_exactly_to_the_logits(hidden):
    torch.manual_seed(0)
    model = AdditiveOCTAttention(feature_dim=8, embedding_dim=16,
                                 attention_dim=8, dropout=0.0,
                                 head_hidden=hidden).eval()
    x = bag(dim=8)
    with torch.inference_mode():
        logits = model(x)[0]
        contributions = model.contributions(x)
    assert contributions.shape == (x.shape[0], 2)
    assert torch.allclose(contributions.sum(0, keepdim=True), logits, atol=1e-6)


def test_contributions_are_signed_both_ways():
    """A constant-sign attribution map would carry no information."""
    torch.manual_seed(0)
    model = AdditiveOCTAttention(feature_dim=8, embedding_dim=16,
                                 attention_dim=8, dropout=0.0,
                                 head_hidden=(16,)).eval()
    with torch.inference_mode():
        c = model.contributions(bag(n=200, dim=8))
    margin = c[:, 1] - c[:, 0]
    assert (margin > 0).any() and (margin < 0).any()


# --- depth 0 must reproduce the manuscript model -----------------------------

def test_linear_head_reproduces_octattention():
    """psi with no hidden layer and no bias == the manuscript linear head."""
    torch.manual_seed(0)
    reference = OCTAttention(feature_dim=8, embedding_dim=16, attention_dim=8,
                             dropout=0.0).eval()
    additive = AdditiveOCTAttention(feature_dim=8, embedding_dim=16,
                                    attention_dim=8, dropout=0.0,
                                    head_hidden=()).eval()
    state = reference.state_dict()
    additive.load_state_dict({
        "embedding.0.weight": state["embedding.0.weight"],
        "embedding.0.bias": state["embedding.0.bias"],
        **{k: v for k, v in state.items() if k.startswith("attention.")},
        "classifier.psi.0.weight": state["classifier.weight"],
    }, strict=False)
    # the manuscript head has a bias; ours does not, so compare without it
    with torch.no_grad():
        reference.classifier.bias.zero_()

    x = bag(dim=8)
    with torch.inference_mode():
        a = reference(x)[0]
        b = additive(x)[0]
    assert float((a - b).abs().max()) < 1e-5


# --- homogeneity: why we compute alpha*psi(h) --------------------------------

def test_bias_free_relu_head_is_positively_homogeneous():
    """psi(alpha*h) == alpha*psi(h), which is what makes the rewrite legal."""
    torch.manual_seed(0)
    head = AdditivePatchHead(input_dim=32, hidden_dims=(16,), dropout=0.0,
                             bias=False).eval()
    h = torch.randn(50, 32)
    alpha = torch.softmax(torch.randn(50), 0)
    with torch.inference_mode():
        direct = head.psi(alpha.unsqueeze(1) * h)
        scaled = alpha.unsqueeze(1) * head.psi(h)
    assert float((direct - scaled).abs().max()) < 1e-6


def test_a_biased_head_collapses_toward_a_constant():
    """Documents the failure the module docstring warns about.

    At realistic bag sizes alpha_i ~ 1/N, so psi(alpha_i h_i) ~ psi(0) for
    every patch and the attribution map carries almost no signal.
    """
    torch.manual_seed(0)
    head = AdditivePatchHead(input_dim=512, hidden_dims=(256,), dropout=0.0,
                             bias=True).eval()
    h = torch.randn(300, 512)
    alpha = torch.softmax(torch.randn(300), 0)
    with torch.inference_mode():
        collapsed = head.psi(alpha.unsqueeze(1) * h)
        constant = head.psi(torch.zeros(1, 512))
        healthy = alpha.unsqueeze(1) * head.psi(h)
    signal = (collapsed - constant).abs().mean() / constant.abs().mean()
    assert float(signal) < 0.1, "expected the biased head to be ~constant"
    # the form we actually use keeps the patches distinguishable
    spread = healthy.std(dim=0).mean() / healthy.abs().mean()
    assert float(spread) > 0.1


# --- a hidden layer makes it a different model -------------------------------

def test_hidden_layer_differs_from_the_classical_model():
    torch.manual_seed(0)
    classical = OCTAttention(feature_dim=8, embedding_dim=16, attention_dim=8,
                             dropout=0.0).eval()
    additive = AdditiveOCTAttention(feature_dim=8, embedding_dim=16,
                                    attention_dim=8, dropout=0.0,
                                    head_hidden=(16,)).eval()
    shared = {k: v for k, v in classical.state_dict().items()
              if not k.startswith("classifier.")}
    additive.load_state_dict({**additive.state_dict(), **shared})
    x = bag(dim=8)
    with torch.inference_mode():
        assert not torch.allclose(classical(x)[0], additive(x)[0], atol=1e-4)


# --- plumbing ----------------------------------------------------------------

def test_output_contract_matches_octattention():
    torch.manual_seed(0)
    model = AdditiveOCTAttention(feature_dim=8, embedding_dim=16,
                                 attention_dim=8, dropout=0.0).eval()
    x = bag(dim=8)
    with torch.inference_mode():
        logits, probabilities, prediction, raw_attention = model(x)
    assert logits.shape == (1, 2)
    assert float(probabilities.sum()) == pytest.approx(1.0, abs=1e-5)
    assert prediction.shape == (1,)
    assert raw_attention.shape == (1, x.shape[0])


def test_batched_input_is_accepted():
    torch.manual_seed(0)
    model = AdditiveOCTAttention(feature_dim=8, embedding_dim=16,
                                 attention_dim=8, dropout=0.0).eval()
    with torch.inference_mode():
        assert model(bag(dim=8).unsqueeze(0))[0].shape == (1, 2)


def test_spatial_mixing_still_works():
    torch.manual_seed(0)
    model = AdditiveOCTAttention(feature_dim=8, embedding_dim=16,
                                 attention_dim=8, dropout=0.0,
                                 spatial_mode="block", block_size=2).eval()
    x = bag(n=12, dim=8)
    coordinates = torch.tensor([(r, c) for r in range(4) for c in range(3)])
    with torch.inference_mode():
        logits = model(x, coordinates)[0]
        contributions = model.contributions(x, coordinates)
    assert torch.allclose(contributions.sum(0, keepdim=True), logits, atol=1e-6)


def test_gradients_reach_the_patch_head():
    torch.manual_seed(0)
    model = AdditiveOCTAttention(feature_dim=8, embedding_dim=16,
                                 attention_dim=8, dropout=0.0,
                                 head_hidden=(16,))
    logits = model(bag(dim=8))[0]
    torch.nn.functional.cross_entropy(logits, torch.tensor([1])).backward()
    grad = model.classifier.psi[0].weight.grad
    assert grad is not None and float(grad.abs().sum()) > 0


def test_manuscript_model_is_untouched():
    """Guard: importing this module must not perturb OCTAttention."""
    torch.manual_seed(0)
    model = OCTAttention(feature_dim=8, embedding_dim=16, attention_dim=8,
                         dropout=0.0).eval()
    with torch.inference_mode():
        assert model(bag(dim=8))[0].shape == (1, 2)
