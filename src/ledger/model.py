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


class SubLayer(nn.Module):
    def __init__(self, cfg, kind: str):
        super().__init__()
        assert kind in ("attn", "mlp")
        self.norm = RMSNorm(cfg.dim)
        if kind == "attn":
            self.fn = CausalSelfAttention(cfg.dim, cfg.n_head, qk_norm=cfg.qk_norm, dropout=cfg.dropout)
        else:
            self.fn = MLP(cfg.dim, cfg.mlp_mult, dropout=cfg.dropout)
        self.res = build_residual(cfg.residual, cfg.dim, cfg)

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

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)

        layers = []
        for _ in range(cfg.n_layer):
            layers.append(SubLayer(cfg, "attn"))
            layers.append(SubLayer(cfg, "mlp"))
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

    def forward(self, idx, targets=None, gamma=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None])

        state = init_state(self.residual, x)
        for sl in self.layers:
            state = sl(state)

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
