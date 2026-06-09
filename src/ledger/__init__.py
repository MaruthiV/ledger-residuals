"""Ledger Residuals — research code.

Public surface kept small on purpose. The five canonical configs all share one
model/data/training path and differ ONLY in the residual op (and a qk_norm flag):

    vanilla        : standard additive pre-norm residual
    hc2            : symmetric two-stream (Hyper-Connections n=2) — controls for extra width
    delta_only     : depth-wise delta-rule erase, single stream — controls for erasure
    suppress_only  : vanilla + QK-norm — the artifact-camp foil (no functional channel)
    ledger         : Deliberation (erasable) + Commitment (protected, decode-only)

See docs/research_plan.md (the science) and docs/CLAUDE.md (working standards).
"""

from .config import ModelConfig, TrainConfig, load_config
from .model import GPT
from .residual_ops import ResidualState

__all__ = ["ModelConfig", "TrainConfig", "load_config", "GPT", "ResidualState"]
