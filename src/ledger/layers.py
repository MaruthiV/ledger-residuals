"""Shared transformer building blocks.

These are IDENTICAL across all five configs — only the residual op differs. Keeping
them shared is what makes the matched-everything comparison (and the degeneracy test)
trustworthy.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    qk_norm=True applies RMSNorm to per-head queries and keys before the dot product.
    This is the `suppress_only` intervention: QK-norm is a standard, effective way to
    suppress massive activations / attention sinks (the 'artifact' camp's tool), used
    here WITHOUT providing any functional commitment channel.
    """

    def __init__(self, dim: int, n_head: int, qk_norm: bool = False, dropout: float = 0.0):
        super().__init__()
        assert dim % n_head == 0, "dim must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.dim = dim
        self.dropout = dropout
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.dim, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, n_head, T, head_dim)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(dim, mult * dim)
        self.proj = nn.Linear(mult * dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))
