"""Go/No-Go #1 harness: train all five configs on commit-vs-revise (matched everything
but the residual op) and measure retention accuracy vs #distractors.

    python -m ledger.compare --steps 2000 --device mps --out outputs/compare

Decision (docs/research_plan.md §17): GO iff `ledger` beats BOTH `hc2` and `delta_only`
with the gap widening as #distractors grows, AND `suppress_only` <= `vanilla`.

Note on this first run: a small FULL-ATTENTION transformer can often retrieve a held
value via attention regardless of residual typing, so curves may be close. That is itself
informative — it tells us whether we need the depth-stress / linear-attention backbone or
the mechanistic massive-activation measurement to separate the configs.
"""
from __future__ import annotations

import argparse
import json
import os
import random

import torch

from .config import ModelConfig, TrainConfig
from .data import CommitReviseTask, make_retention_batch
from .train import train
from .utils import pick_device

# commit-vs-revise difficulty (shared by training and eval so the vocab matches)
TASK_KW = dict(n_slots=6, n_vals=12, min_events=8, max_events=24)

# matched-everything base; only the residual op (+ qk_norm / ledger knobs) varies
BASE_MODEL = dict(n_layer=4, n_head=4, dim=128, block_size=128, mlp_mult=4)
PRESETS = {
    "vanilla":       dict(residual="vanilla"),
    "hc2":           dict(residual="hc2"),
    "delta_only":    dict(residual="delta_only", tie_delta_gates=True),
    "suppress_only": dict(residual="vanilla", qk_norm=True),
    "ledger":        dict(residual="ledger", gamma=0.0, commit_bias=3.0),  # decode-from-C, gate starts open
}
K_SWEEP = [0, 1, 2, 4, 8, 12, 16, 24, 32]


def build_config(name: str, steps: int, device: str, seed: int) -> TrainConfig:
    model = ModelConfig(**BASE_MODEL, **PRESETS[name])
    return TrainConfig(
        model=model,
        max_steps=steps,
        device=device,
        batch_size=64,
        lr=3e-4,
        log_every=max(100, steps // 10),
        seed=seed,
        warmup_gamma_steps=0,   # ledger decodes from C from step 0 (commit gate starts open) — no D-crutch to lose
        commit_sparsity=0.0,    # off until decode-from-C is confirmed to train; selectivity studied next
        **TASK_KW,
    )


@torch.no_grad()
def retention_accuracy(model, task, k, n_examples, block_size, device, eval_seed=12345):
    rng = random.Random(eval_seed + k)  # same eval examples per k across all configs
    model.eval()
    correct = total = 0
    done = 0
    while done < n_examples:
        b = min(256, n_examples - done)
        ids, tgt = make_retention_batch(task, k, b, block_size, rng)
        ids, tgt = ids.to(device), tgt.to(device)
        logits, _ = model(ids, tgt)
        mask = tgt != -100
        pred = logits.argmax(-1)
        correct += (pred[mask] == tgt[mask]).sum().item()
        total += int(mask.sum().item())
        done += b
    return correct / max(total, 1)


def run(steps: int, device: str, out: str, n_eval: int, only, seed: int):
    device = pick_device(device)
    os.makedirs(out, exist_ok=True)
    names = only or list(PRESETS)
    task = CommitReviseTask(**TASK_KW)  # shared by all configs so the vocab matches

    results = {}
    for name in names:
        print(f"\n{'=' * 60}\n=== TRAIN  {name}\n{'=' * 60}")
        cfg = build_config(name, steps, device, seed)
        model, history = train(cfg)
        ret = {
            k: retention_accuracy(model, task, k, n_eval, cfg.model.block_size, device)
            for k in K_SWEEP
        }
        results[name] = {
            "final_train_acc": history[-1]["acc"],
            "retention": ret,
            "commit_per_layer": history[-1].get("commit_per_layer"),
        }
        print(f"[{name}] retention " + " ".join(f"k{k}={ret[k]:.3f}" for k in K_SWEEP))

    _summary_table(results)
    with open(os.path.join(out, "results.json"), "w") as fh:
        json.dump({"steps": steps, "k_sweep": K_SWEEP, "results": results}, fh, indent=2)
    _plot(results, out)
    print(f"\nSaved results.json and accuracy_vs_distractors.png to {out}/")
    return results


def _summary_table(results):
    print(f"\n{'=' * 70}\nSUMMARY — retention accuracy vs #distractors\n{'=' * 70}")
    print("config".ljust(15) + "".join(f"k={k}".rjust(7) for k in K_SWEEP))
    for name, res in results.items():
        print(name.ljust(15) + "".join(f"{res['retention'][k]:.3f}".rjust(7) for k in K_SWEEP))
    if "ledger" in results and results["ledger"]["commit_per_layer"]:
        print("\nledger commit-gate per sublayer:", [round(x, 2) for x in results["ledger"]["commit_per_layer"]])


def _plot(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, res in results.items():
        accs = [res["retention"][k] for k in K_SWEEP]
        ax.plot(K_SWEEP, accs, marker="o", label=name)
    ax.set_xlabel("# distractor events after the defining SET")
    ax.set_ylabel("query accuracy (commitment retention)")
    ax.set_title("commit-vs-revise: retention vs distractors")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "accuracy_vs_distractors.png"), dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="outputs/compare")
    ap.add_argument("--n-eval", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default=None, help="comma-separated subset of configs")
    args = ap.parse_args()
    only = args.only.split(",") if args.only else None
    run(args.steps, args.device, args.out, args.n_eval, only, args.seed)


if __name__ == "__main__":
    main()
