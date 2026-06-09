"""Degeneracy guards (docs/research_plan.md §5, Proposition 3).

These prove the implementation has an exact vanilla-equivalent corner, so any future
change that breaks the shared plumbing (norm / attention / mlp / embed / unembed) is
caught immediately. The two-stream `ledger` op and the decoupled `delta` op must both
reduce to a standard pre-norm transformer at their degenerate settings.
"""
import torch

from ledger.config import ModelConfig
from ledger.model import GPT


def _cfg(residual: str, **kw) -> ModelConfig:
    base = dict(vocab_size=32, block_size=16, n_layer=3, n_head=2, dim=32, mlp_mult=2, dropout=0.0)
    base.update(kw)
    return ModelConfig(residual=residual, **base)


def _copy_shared_params(src: GPT, dst: GPT) -> None:
    """Copy every param that is NOT inside a residual op (so only the residual differs)."""
    s, d = src.state_dict(), dst.state_dict()
    for key, val in s.items():
        if ".res." not in key and key in d and d[key].shape == val.shape:
            d[key] = val.clone()
    dst.load_state_dict(d, strict=True)


def _max_logit_diff(model_a: GPT, model_b: GPT, vocab: int, B: int = 3, T: int = 12) -> float:
    model_a.eval()
    model_b.eval()
    x = torch.randint(0, vocab, (B, T))
    with torch.no_grad():
        la, _ = model_a(x)
        lb, _ = model_b(x)
    return (la - lb).abs().max().item()


def test_ledger_reduces_to_vanilla():
    torch.manual_seed(0)
    vanilla = GPT(_cfg("vanilla"))
    ledger = GPT(_cfg("ledger"))
    _copy_shared_params(vanilla, ledger)
    ledger.vanillaize_residuals()  # pins gates: beta_e->0, beta_w->1, g=I, c->0, gamma=1
    diff = _max_logit_diff(vanilla, ledger, vocab=32)
    assert diff < 1e-3, f"ledger != vanilla at degenerate corner (max logit diff = {diff:.2e})"


def test_delta_reduces_to_vanilla():
    torch.manual_seed(1)
    vanilla = GPT(_cfg("vanilla"))
    delta = GPT(_cfg("delta_only", tie_delta_gates=False))  # decoupled gates can recover addition
    _copy_shared_params(vanilla, delta)
    delta.vanillaize_residuals()
    diff = _max_logit_diff(vanilla, delta, vocab=32)
    assert diff < 1e-3, f"delta != vanilla at degenerate corner (max logit diff = {diff:.2e})"
