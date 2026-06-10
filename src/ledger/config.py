"""Config dataclasses + a tiny YAML loader (no framework, no inheritance)."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 64          # set at runtime from the task
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    dim: int = 128
    mlp_mult: int = 4
    dropout: float = 0.0
    # residual selection
    residual: str = "vanilla"     # vanilla | hc2 | delta_only | ledger
    qk_norm: bool = False         # suppress_only = (residual=vanilla, qk_norm=True)
    tie_delta_gates: bool = True  # delta_only baseline ties erase/write into one gate
    gamma: float = 0.0            # ledger decode mix: norm(C + gamma*D)
    commit_budget: Optional[float] = None
    # Depth-scheduled commit-gate bias: commit ~nothing early (deliberate), commit late (answer ready).
    # Fixes dilution-inside-C (gate-open-everywhere accumulates pre-answer deliberation into C).
    commit_bias_early: float = -4.0
    commit_bias_late: float = 4.0
    gate_rank: int = 64           # low-rank factor for ledger gates (keeps params ~vanilla; 0 = full-rank)
    init_vanilla: bool = True     # start near vanilla (stability + anti-"just HC")


@dataclass
class TrainConfig:
    # commit-vs-revise task
    n_slots: int = 4
    n_vals: int = 8
    min_events: int = 6
    max_events: int = 20
    p_set: float = 0.45
    p_noise: float = 0.30
    p_query: float = 0.25
    # optimization
    lr: float = 3e-4
    weight_decay: float = 0.1
    batch_size: int = 64
    max_steps: int = 2000
    grad_clip: float = 1.0
    warmup_gamma_steps: int = 0   # anneal ledger gamma 1->0 over these steps (0 = use model.gamma fixed)
    commit_sparsity: float = 0.0  # alpha * mean(commit gate): the ledger gate-collapse guard
    # runtime
    seed: int = 0
    device: str = "auto"          # auto | cpu | mps | cuda
    log_every: int = 100
    out_dir: str = "outputs/run"
    model: ModelConfig = field(default_factory=ModelConfig)


@dataclass
class LMConfig:
    """Language-model training config (the headline mechanistic run)."""
    data: str = "pseudo"          # "pseudo" (local smoke) | "fineweb" (Modal)
    tokenizer: str = "bytes"      # "bytes" | "gpt2"
    max_tokens: int = 2_000_000
    val_tokens: int = 200_000
    lr: float = 3e-4
    weight_decay: float = 0.1
    batch_size: int = 32
    max_steps: int = 2000
    warmup_steps: int = 100
    grad_clip: float = 1.0
    commit_sparsity: float = 0.02
    seed: int = 0
    device: str = "auto"
    log_every: int = 100
    eval_every: int = 500
    out_dir: str = "outputs/lm"
    model: ModelConfig = field(default_factory=ModelConfig)


def _filter(cls, d: dict) -> dict:
    valid = {f.name for f in fields(cls)}
    unknown = set(d) - valid
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return {k: v for k, v in d.items() if k in valid}


def load_config(path: str) -> TrainConfig:
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    model_raw = raw.pop("model", {}) or {}
    return TrainConfig(model=ModelConfig(**_filter(ModelConfig, model_raw)), **_filter(TrainConfig, raw))


def load_lm_config(path: str) -> LMConfig:
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    model_raw = raw.pop("model", {}) or {}
    return LMConfig(model=ModelConfig(**_filter(ModelConfig, model_raw)), **_filter(LMConfig, raw))
