"""Residual ops — the only thing that differs across the five configs.

Interface (per sublayer, called twice per transformer block):
    read(state)           -> premix tensor that the block then RMSNorms and feeds to the sublayer
    update(state, u, y)    -> new ResidualState, given normalized input `u` and sublayer output `y`

Model-level helpers (depend on residual type, not on per-block params):
    init_state(name, x)            -> ResidualState at the embedding
    decode_state(name, state, g)   -> pre-final-norm tensor to unembed

Design note (degeneracy): the Deliberation update uses DECOUPLED erase/write gates
(beta_e, beta_w), not a single tied beta. With a tied gate, beta->0 *freezes* the
stream (identity), which is NOT vanilla. Decoupled gates let beta_e->0, beta_w->1
recover pure addition  D <- D + y  — the exact vanilla-equivalent corner. This also
mirrors the Gated-DeltaNet-2 "decouple erase and write" move (docs/research_plan.md §5).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# sigmoid(±_BIG) ≈ 0/1 to ~1e-13 — used ONLY to pin gates for the degeneracy test.
_BIG = 30.0


@dataclass
class ResidualState:
    """Residual state threaded through the network.

    primary   : h (vanilla/delta/suppress) | D=Deliberation (ledger) | stream-0 (hc2)
    secondary : None | C=Commitment, protected (ledger) | stream-1 (hc2)
    """
    primary: torch.Tensor
    secondary: Optional[torch.Tensor] = None


def _set_identity(linear: nn.Linear) -> None:
    with torch.no_grad():
        w = linear.weight
        linear.weight.copy_(torch.eye(w.shape[0], w.shape[1], device=w.device, dtype=w.dtype))
        if linear.bias is not None:
            linear.bias.zero_()


# --------------------------------------------------------------------------------------
# vanilla
# --------------------------------------------------------------------------------------
class VanillaResidual(nn.Module):
    """h <- h + sublayer(norm(h))."""
    n_streams = 1

    def read(self, s: ResidualState) -> torch.Tensor:
        return s.primary

    def update(self, s: ResidualState, u: torch.Tensor, y: torch.Tensor) -> ResidualState:
        return ResidualState(s.primary + y)


# --------------------------------------------------------------------------------------
# hc2 — symmetric two-stream (minimal static Hyper-Connections n=2)
# --------------------------------------------------------------------------------------
class HyperResidual(nn.Module):
    """Symmetric two-stream residual; both streams summed before the head.

    Minimal STATIC Hyper-Connections n=2 (read-mix + 2x2 width-mix + output distribution).
    Purpose: a control for 'extra width / two streams' with NO protect/erase typing and
    NO decode-from-one. Full dynamic HC is deferred (docs/TASKS.md T2.8); this faithful-
    enough version is what Go/No-Go #1 needs (does asymmetry beat symmetry?).
    """
    n_streams = 2

    def __init__(self, dim: int):
        super().__init__()
        self.read_mix = nn.Parameter(torch.tensor([0.5, 0.5]))   # how streams combine into the sublayer input
        self.width_mix = nn.Parameter(torch.eye(2))              # 2x2 cross-stream mixing
        self.out_mix = nn.Parameter(torch.tensor([0.5, 0.5]))    # how the sublayer output is distributed to streams

    def read(self, s: ResidualState) -> torch.Tensor:
        return self.read_mix[0] * s.primary + self.read_mix[1] * s.secondary

    def update(self, s: ResidualState, u: torch.Tensor, y: torch.Tensor) -> ResidualState:
        s0, s1 = s.primary, s.secondary
        m = self.width_mix
        n0 = m[0, 0] * s0 + m[0, 1] * s1 + self.out_mix[0] * y
        n1 = m[1, 0] * s0 + m[1, 1] * s1 + self.out_mix[1] * y
        return ResidualState(n0, n1)


# --------------------------------------------------------------------------------------
# delta_only — depth-wise delta-rule erase, single stream (Deep-Delta-style baseline)
# --------------------------------------------------------------------------------------
class DeltaResidual(nn.Module):
    """h <- h - beta_e ⊙ (kᵀh) k + beta_w ⊙ g(y).

    tie_gates=True  -> beta_w = beta_e (a single erase-then-write gate; the delta_only baseline).
    tie_gates=False -> decoupled gates (can reduce to vanilla addition; used by the degeneracy test).
    """
    n_streams = 1

    def __init__(self, dim: int, tie_gates: bool = True):
        super().__init__()
        self.tie_gates = tie_gates
        self.w_e = nn.Linear(dim, dim)            # erase-gate logits (channel-wise)
        self.w_w = nn.Linear(dim, dim)            # write-gate logits
        self.w_k = nn.Linear(dim, dim, bias=False)  # erase direction
        self.g = nn.Linear(dim, dim, bias=False)    # write projection of the sublayer output
        self._init_stable()

    def read(self, s: ResidualState) -> torch.Tensor:
        return s.primary

    def update(self, s: ResidualState, u: torch.Tensor, y: torch.Tensor) -> ResidualState:
        h = s.primary
        beta_e = torch.sigmoid(self.w_e(u))
        beta_w = beta_e if self.tie_gates else torch.sigmoid(self.w_w(u))
        k = F.normalize(self.w_k(u), dim=-1)
        erase = beta_e * (k * (k * h).sum(-1, keepdim=True))
        write = beta_w * self.g(y)
        return ResidualState(h - erase + write)

    @torch.no_grad()
    def _init_stable(self):
        # Start near vanilla but trainable: gates mild (gradients flow), write passes through.
        _set_identity(self.g)
        self.w_e.weight.zero_(); self.w_e.bias.fill_(-3.0)   # beta_e = sigmoid(-3) ≈ 0.047 (low erase)
        self.w_w.weight.zero_(); self.w_w.bias.fill_(+3.0)   # beta_w = sigmoid(+3) ≈ 0.953 (high write)

    @torch.no_grad()
    def set_vanilla_corner(self):
        assert not self.tie_gates, "vanilla corner requires decoupled gates (tie_gates=False)"
        self.w_e.weight.zero_(); self.w_e.bias.fill_(-_BIG)  # beta_e ≈ 0  (no erase)
        self.w_w.weight.zero_(); self.w_w.bias.fill_(+_BIG)  # beta_w ≈ 1  (full write)
        _set_identity(self.g)                                # write = y  =>  h <- h + y


# --------------------------------------------------------------------------------------
# ledger — Deliberation (erasable) + Commitment (protected, decode-only)
# --------------------------------------------------------------------------------------
class LedgerResidual(nn.Module):
    """Asymmetric two-stream residual.

    Read:        u_premix = D + lam ⊙ C            (C is read-only here)
    Deliberation: D <- D - beta_e⊙(kᵀD)k + beta_w⊙g(y)   (decoupled delta-rule; can erase)
    Commitment:   C <- C + c · P(D)                (append-only; gated; no erase term)
                  optional mass budget: C <- C · min(1, rho/‖C‖)
    Decode (model level): norm(C + gamma·D).
    """
    n_streams = 2

    def __init__(self, dim: int, commit_budget: Optional[float] = None, init_vanilla: bool = True):
        super().__init__()
        # Deliberation gates (decoupled erase/write)
        self.w_e = nn.Linear(dim, dim)
        self.w_w = nn.Linear(dim, dim)
        self.w_k = nn.Linear(dim, dim, bias=False)
        self.g = nn.Linear(dim, dim, bias=False)
        # Commitment
        self.w_c = nn.Linear(dim, 1)                 # scalar commit gate
        self.P = nn.Linear(dim, dim, bias=False)     # promotion D -> C
        self.lam = nn.Parameter(torch.zeros(dim))    # read coupling C -> D input (init 0)
        self.commit_budget = commit_budget
        if init_vanilla:
            self._init_stable()

    def read(self, s: ResidualState) -> torch.Tensor:
        return s.primary + self.lam * s.secondary

    def update(self, s: ResidualState, u: torch.Tensor, y: torch.Tensor) -> ResidualState:
        D, C = s.primary, s.secondary
        beta_e = torch.sigmoid(self.w_e(u))
        beta_w = torch.sigmoid(self.w_w(u))
        k = F.normalize(self.w_k(u), dim=-1)
        D = D - beta_e * (k * (k * D).sum(-1, keepdim=True)) + beta_w * self.g(y)
        c = torch.sigmoid(self.w_c(u))               # (B, T, 1)
        C = C + c * self.P(D)
        if self.commit_budget is not None:
            scale = torch.clamp(self.commit_budget / (C.norm(dim=-1, keepdim=True) + 1e-6), max=1.0)
            C = C * scale
        return ResidualState(D, C)

    def commit_rate(self, u: torch.Tensor) -> torch.Tensor:
        """Mean commit-gate activation for logging / the (future) commit-sparsity penalty."""
        return torch.sigmoid(self.w_c(u)).mean()

    @torch.no_grad()
    def _init_stable(self):
        # Start near vanilla but trainable: D adds (mostly) like a residual, C commits little.
        _set_identity(self.g)
        _set_identity(self.P)
        self.w_e.weight.zero_(); self.w_e.bias.fill_(-3.0)   # low erase
        self.w_w.weight.zero_(); self.w_w.bias.fill_(+3.0)   # high write
        self.w_c.weight.zero_(); self.w_c.bias.fill_(-3.0)   # commit gate starts low but trainable
        self.lam.zero_()

    @torch.no_grad()
    def set_vanilla_corner(self):
        # Hard pin for the degeneracy test: D-update -> pure addition, C stays empty.
        self.w_e.weight.zero_(); self.w_e.bias.fill_(-_BIG)  # beta_e ≈ 0
        self.w_w.weight.zero_(); self.w_w.bias.fill_(+_BIG)  # beta_w ≈ 1
        _set_identity(self.g)                                # write = y => D <- D + y
        self.w_c.weight.zero_(); self.w_c.bias.fill_(-_BIG)  # c ≈ 0 => C stays 0
        _set_identity(self.P)
        self.lam.zero_()
        # With gamma=1 at decode: norm(C + 1·D) = norm(D) = vanilla h.


# --------------------------------------------------------------------------------------
# factory + state helpers
# --------------------------------------------------------------------------------------
def build_residual(name: str, dim: int, cfg) -> nn.Module:
    if name == "vanilla":
        return VanillaResidual()
    if name == "hc2":
        return HyperResidual(dim)
    if name == "delta_only":
        return DeltaResidual(dim, tie_gates=cfg.tie_delta_gates)
    if name == "ledger":
        return LedgerResidual(dim, commit_budget=cfg.commit_budget, init_vanilla=cfg.init_vanilla)
    raise ValueError(f"unknown residual op: {name!r}")


def init_state(name: str, x: torch.Tensor) -> ResidualState:
    if name in ("vanilla", "delta_only"):
        return ResidualState(x)
    if name == "hc2":
        return ResidualState(x, x.clone())          # symmetric: both streams start at the embedding
    if name == "ledger":
        return ResidualState(x, torch.zeros_like(x))  # D = embed, C = 0 (nothing committed yet)
    raise ValueError(f"unknown residual op: {name!r}")


def decode_state(name: str, s: ResidualState, gamma: float) -> torch.Tensor:
    if name in ("vanilla", "delta_only"):
        return s.primary
    if name == "hc2":
        return s.primary + s.secondary              # streams summed before the head
    if name == "ledger":
        return s.secondary + gamma * s.primary      # decode from Commitment (+ gamma·Deliberation)
    raise ValueError(f"unknown residual op: {name!r}")
