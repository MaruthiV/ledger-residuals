"""First mechanistic experiment: residual-stream dilution & activation profile vs depth.

    python -m ledger.mechanism --steps 800 --device mps --out outputs/mechanism

Tests the dilution backbone of the thesis (research_plan.md §5, Prop 1): a pre-norm
residual stream accumulates O(L) content, so its norm grows with depth and early-written
content is relatively diluted at the readout — which the model would compensate for with
massive activations (high max/median ratio). We measure, per sublayer:
  - mean token L2 norm of the residual stream (does it grow ~sqrt(depth)?)
  - max/median activation ratio (the massive-activation proxy, Sun et al. 2024)
and compare vanilla / suppress_only / ledger (whose decoded Commitment stream C should
show controlled norm growth).

NOTE: at this toy scale on synthetic data, dramatic massive activations are not expected;
the clean, scale-free signal here is the NORM-GROWTH geometry. Real MA emergence needs the
language-model + scale run on Modal (research_plan.md §7.3).
"""
from __future__ import annotations

import argparse
import json
import os
import random

import torch

from .config import ModelConfig, TrainConfig
from .data import CommitReviseTask, make_batch
from .train import train
from .utils import pick_device

TASK_KW = dict(n_slots=6, n_vals=12, min_events=8, max_events=24)
CONFIGS = {
    "vanilla": dict(residual="vanilla"),
    "suppress_only": dict(residual="vanilla", qk_norm=True),
    "ledger": dict(residual="ledger", gamma=0.0),
}
DEPTHS = [4, 8, 12]


def build(name, depth, steps, device, seed):
    model = ModelConfig(n_layer=depth, n_head=4, dim=128, block_size=128, mlp_mult=4, **CONFIGS[name])
    return TrainConfig(model=model, max_steps=steps, device=device, batch_size=64, lr=3e-4,
                       log_every=max(200, steps // 3), seed=seed, **TASK_KW)


@torch.no_grad()
def measure(model, task, device, block_size, seed=999):
    rng = random.Random(seed)
    ids, _ = make_batch(task, 256, block_size, rng)
    model.eval()
    model(ids.to(device), record_trace=True)
    return model.aux["trace"]


def run(steps, device, out, seed):
    device = pick_device(device)
    os.makedirs(out, exist_ok=True)
    task = CommitReviseTask(**TASK_KW)
    results = {}
    for name in CONFIGS:
        for depth in DEPTHS:
            print(f"\n=== {name} depth={depth} ===")
            cfg = build(name, depth, steps, device, seed)
            model, _ = train(cfg)
            trace = measure(model, task, device, cfg.model.block_size)
            results[f"{name}_d{depth}"] = trace
            print(f"[{name} d{depth}] norm:", [round(t["primary_norm"], 1) for t in trace])
            print(f"[{name} d{depth}] max/med:", [round(t["primary_maxmed"], 1) for t in trace])

    with open(os.path.join(out, "mechanism.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    _plot(results, out)
    print(f"\nsaved mechanism.json + plots to {out}/")


def _plot(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    deepest = max(DEPTHS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name in CONFIGS:
        trace = results[f"{name}_d{deepest}"]
        xs = [i / (len(trace) - 1) for i in range(len(trace))]
        axes[0].plot(xs, [t["primary_norm"] for t in trace], marker=".", label=f"{name}")
        if "secondary_norm" in trace[0]:
            axes[0].plot(xs, [t["secondary_norm"] for t in trace], marker="x", linestyle="--",
                         label=f"{name} (C)")
        axes[1].plot(xs, [t["primary_maxmed"] for t in trace], marker=".", label=name)
    axes[0].set(title=f"residual norm vs depth (n_layer={deepest})", xlabel="fractional depth",
                ylabel="mean token L2 norm")
    axes[1].set(title="max/median activation ratio vs depth", xlabel="fractional depth",
                ylabel="max/median")
    for ax in axes:
        ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "residual_profile.png"), dpi=120)

    fig2, ax = plt.subplots(figsize=(7, 5))
    for name in CONFIGS:
        ax.plot(DEPTHS, [results[f"{name}_d{d}"][-1]["primary_maxmed"] for d in DEPTHS], marker="o", label=name)
    ax.set(title="last-sublayer max/median vs model depth", xlabel="n_layer", ylabel="max/median")
    ax.legend(); ax.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(out, "maxmed_vs_depth.png"), dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="outputs/mechanism")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.steps, a.device, a.out, a.seed)


if __name__ == "__main__":
    main()
