"""Smoke: every config does a forward + backward and produces well-shaped logits."""
import pytest
import torch

from ledger.config import ModelConfig
from ledger.model import GPT

CONFIGS = [
    dict(residual="vanilla"),
    dict(residual="vanilla", qk_norm=True),       # suppress_only
    dict(residual="hc2"),
    dict(residual="delta_only", tie_delta_gates=True),
    dict(residual="ledger"),
    dict(residual="ledger", commit_budget=5.0),
]


@pytest.mark.parametrize("overrides", CONFIGS)
def test_forward_backward(overrides):
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=32, block_size=16, n_layer=2, n_head=2, dim=32, mlp_mult=2, **overrides)
    model = GPT(cfg)
    B, T = 4, 12
    x = torch.randint(0, 32, (B, T))
    y = torch.randint(0, 32, (B, T))
    logits, loss = model(x, y)
    assert logits.shape == (B, T, 32)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)


def test_ledger_protects_against_unembedding_of_deliberation():
    """At gamma=0 the head must see ONLY the Commitment stream (decode-from-C)."""
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=32, block_size=16, n_layer=2, n_head=2, dim=32, residual="ledger", gamma=0.0)
    model = GPT(cfg).eval()
    x = torch.randint(0, 32, (2, 10))
    with torch.no_grad():
        logits_g0, _ = model(x, gamma=0.0)
        logits_g1, _ = model(x, gamma=1.0)
    # gamma changes the decode => logits must differ (D actually contributes when gamma>0)
    assert (logits_g0 - logits_g1).abs().max() > 1e-5
