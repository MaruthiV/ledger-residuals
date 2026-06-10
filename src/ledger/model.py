"""GPT backbone with a pluggable residual op.

The model is a flat sequence of SubLayers (attn, mlp, attn, mlp, ...) so that the
per-sublayer residual indexing matches the method in docs/research_plan.md §5. The
ONLY thing that varies across configs is `SubLayer.res`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm, CausalSelfAttention, MLP
from .residual_ops import build_residual, init_state, decode_state


def _act_stats(x) -> dict:
    """Massive-activation diagnostics for one stream (Sun et al. 2024 conventions).

    x: (B, T, d). Returns norm-confounded AND norm-independent measures:
    - norm        : mean per-token L2 norm
    - maxmed      : mean (max|x| / median|x|)            -- ratio (confounded by norm growth)
    - absmax      : mean max|x|                           -- RAW magnitude (un-confounded)
    - kurtosis    : mean excess kurtosis over dims        -- heavy-tail/outlier measure (scale-free)
    - feat_maxmed : max_dim(mean_tokens|x|) / median      -- FIXED-dimension signature (the true Sun MA:
                    a few input-agnostic dims with persistently large magnitude)
    - n_out50     : mean #dims with |x| >= 50x token median
    - n_sun       : mean #dims 'massive' by Sun's absolute rule (|x|>100 AND >=1000x median)
    """
    absx = x.abs()
    med = absx.median(-1, keepdim=True).values  # (..., 1)
    maxmed = (absx.amax(-1) / (med.squeeze(-1) + 1e-6)).mean().item()
    n_out50 = (absx >= 50.0 * med).float().sum(-1).mean().item()
    n_sun = ((absx > 100.0) & (absx >= 1000.0 * med)).float().sum(-1).mean().item()
    absmax = absx.amax(-1).mean().item()
    # excess kurtosis over channels, per token, averaged
    xc = x - x.mean(-1, keepdim=True)
    m2 = xc.pow(2).mean(-1)
    kurt = (xc.pow(4).mean(-1) / (m2.pow(2) + 1e-8) - 3.0).mean().item()
    # fixed-dimension signature: per-channel mean |activation| across the batch
    feat = absx.reshape(-1, absx.shape[-1]).mean(0)  # (d,)
    feat_maxmed = (feat.amax() / (feat.median() + 1e-6)).item()
    return {"norm": x.norm(dim=-1).mean().item(), "maxmed": maxmed, "absmax": absmax,
            "kurtosis": kurt, "feat_maxmed": feat_maxmed, "n_out50": n_out50, "n_sun": n_sun}


def _state_stats(state) -> dict:
    """Per-sublayer residual diagnostics for primary (and Commitment, if present)."""
    rec = {f"primary_{k}": v for k, v in _act_stats(state.primary).items()}
    if state.secondary is not None:
        rec.update({f"secondary_{k}": v for k, v in _act_stats(state.secondary).items()})
    return rec


class SubLayer(nn.Module):
    def __init__(self, cfg, kind: str, depth_frac: float = 1.0):
        super().__init__()
        assert kind in ("attn", "mlp")
        self.norm = RMSNorm(cfg.dim)
        if kind == "attn":
            self.fn = CausalSelfAttention(cfg.dim, cfg.n_head, qk_norm=cfg.qk_norm, dropout=cfg.dropout)
        else:
            self.fn = MLP(cfg.dim, cfg.mlp_mult, dropout=cfg.dropout)
        self.res = build_residual(cfg.residual, cfg.dim, cfg, depth_frac=depth_frac)

    def forward(self, state):
        premix = self.res.read(state)
        u = self.norm(premix)
        y = self.fn(u)
        return self.res.update(state, u, y)


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.residual = cfg.residual
        self.gamma = cfg.gamma  # decode mix for ledger: norm(C + gamma*D); annealed during training
        self.aux = {}           # per-forward diagnostics (e.g. commit-gate stats for ledger)

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)

        layers = []
        n_sub = 2 * cfg.n_layer
        for i in range(cfg.n_layer):
            layers.append(SubLayer(cfg, "attn", depth_frac=(2 * i) / max(1, n_sub - 1)))
            layers.append(SubLayer(cfg, "mlp", depth_frac=(2 * i + 1) / max(1, n_sub - 1)))
        self.layers = nn.ModuleList(layers)

        self.final_norm = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        # Generic init for the shared transformer params, THEN restore residual-op stable init
        # (apply() would otherwise clobber the identity/gate init done in the residual constructors).
        self.apply(self._init_weights)
        for sl in self.layers:
            if hasattr(sl.res, "_init_stable"):
                sl.res._init_stable()

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def vanillaize_residuals(self):
        """Pin every residual op to the exact vanilla-equivalent corner (degeneracy test)."""
        for sl in self.layers:
            if hasattr(sl.res, "set_vanilla_corner"):
                sl.res.set_vanilla_corner()
        self.gamma = 1.0

    def forward(self, idx, targets=None, gamma=None, record_trace=False):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None])

        state = init_state(self.residual, x)
        trace = [] if record_trace else None
        for sl in self.layers:
            state = sl(state)
            if record_trace:
                trace.append(_state_stats(state))

        commits = [sl.res.last_commit for sl in self.layers
                   if getattr(sl.res, "last_commit", None) is not None]
        self.aux = (
            {"commit_rate": torch.stack(commits).mean(),
             "commit_per_layer": [c.item() for c in commits]}
            if commits else {}
        )
        if record_trace:
            self.aux["trace"] = trace

        g = self.gamma if gamma is None else gamma
        h = self.final_norm(decode_state(self.residual, state, g))
        logits = self.head(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        return logits, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
