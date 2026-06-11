"""Post-training probes (#1 characterize the relocated outliers, #2 quantization).

#1: Are the outliers the *canonical* massive activations (Sun et al. 2024) — i.e. a few FIXED
    feature dimensions that persist across layers and concentrate on the start/delimiter token —
    and are they confined to the scratch stream `D` while the decoded `C` stays clean?
#2: Does the outlier structure matter for weight PTQ perplexity? (Weak test; MAs are an activation
    phenomenon, so a null here is expected and reported honestly.)
"""
from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def _capture_perdim(model, ids):
    """Per-sublayer per-dimension mean |activation| (over batch+positions), plus position-0 only."""
    full, full0, sec, sec0 = [], [], [], []

    def hook(_module, _inp, out):
        p = out.primary
        full.append(p.abs().mean(dim=(0, 1)).float().cpu())
        full0.append(p[:, 0, :].abs().mean(0).float().cpu())
        if out.secondary is not None:
            s = out.secondary
            sec.append(s.abs().mean(dim=(0, 1)).float().cpu())
            sec0.append(s[:, 0, :].abs().mean(0).float().cpu())

    handles = [sl.register_forward_hook(hook) for sl in model.layers]
    model.eval()
    model(ids)
    for h in handles:
        h.remove()
    return full, full0, sec, sec0


def _fixed_dim_report(perdim, perdim_pos0):
    """Characterize the fixed-dimension MA signature of one channel across depth."""
    if not perdim:
        return None
    M = torch.stack(perdim)        # (L, d) mean |act| per dim per layer
    M0 = torch.stack(perdim_pos0)  # (L, d) at position 0
    med = M.median(dim=1).values   # (L,)
    top_dim = M.argmax(dim=1)      # (L,) top outlier dim per layer
    ratio = (M.max(dim=1).values / (med + 1e-6))     # per-layer feat_maxmed
    dom = M.mean(0).argmax().item()                  # dim with highest mean magnitude across depth
    peak_layer = int(M[:, dom].argmax().item())
    # BOS concentration of the dominant dim at its peak layer: pos-0 magnitude / overall magnitude
    bos_ratio = float(M0[peak_layer, dom] / (M[peak_layer, dom] + 1e-6))
    return {
        "dominant_dim": dom,
        "persistence": round((top_dim == dom).float().mean().item(), 2),  # frac of layers dom is THE top dim
        "peak_layer": peak_layer,
        "peak_feat_ratio": round(float(ratio.max()), 1),
        "bos_concentration": round(bos_ratio, 2),  # >1 => outlier concentrates on the start token
        "feat_ratio_by_layer": [round(x, 1) for x in ratio.tolist()],
    }


@torch.no_grad()
def _weight_ptq_loss(model, eval_fn, bits):
    """Symmetric per-output-channel weight fake-quant -> val loss. Restores weights after."""
    saved = {k: v.clone() for k, v in model.state_dict().items()}
    qmax = 2 ** (bits - 1) - 1
    for m in model.modules():
        if isinstance(m, nn.Linear):
            w = m.weight.data
            scale = w.abs().amax(dim=1, keepdim=True) / qmax + 1e-8
            m.weight.data = torch.round(w / scale).clamp(-qmax - 1, qmax) * scale
    loss = eval_fn()
    model.load_state_dict(saved)
    return loss


def run_probe(model, val_data, cfg, device, gen, evaluate):
    """evaluate: zero-arg callable returning current val loss (closure over model/val_data)."""
    ids, _ = val_data.batch(min(getattr(cfg, "batch_size", 16), 16), device, gen)
    full, full0, sec, sec0 = _capture_perdim(model, ids)
    is_ledger = model.residual == "ledger"
    decoded_full, decoded0 = (sec, sec0) if is_ledger else (full, full0)
    return {
        "decoded_channel": _fixed_dim_report(decoded_full, decoded0),
        "scratch_channel": _fixed_dim_report(full, full0) if is_ledger else None,
        "ptq_val_loss": {
            "fp": evaluate(),
            "w8": _weight_ptq_loss(model, evaluate, 8),
            "w4": _weight_ptq_loss(model, evaluate, 4),
        },
    }
