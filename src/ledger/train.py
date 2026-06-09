"""Device-agnostic training entrypoint.

Runs identically on CPU (smoke test) and GPU (Modal). No training logic lives in
modal_app.py — it just calls train(cfg). See docs/CLAUDE.md (Modal rules).

    python -m ledger.train --config configs/toy_smoke.yaml          # CPU smoke
    modal run modal_app.py --config configs/toy.yaml                # GPU on Modal
"""
from __future__ import annotations

import argparse
import random

import torch

from .config import TrainConfig, load_config
from .data import CommitReviseTask, make_batch
from .model import GPT
from .utils import pick_device, set_seed, human


def query_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    mask = targets != -100
    if mask.sum() == 0:
        return float("nan")
    pred = logits.argmax(-1)
    return (pred[mask] == targets[mask]).float().mean().item()


def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    rng = random.Random(cfg.seed)

    task = CommitReviseTask(
        cfg.n_slots, cfg.n_vals, cfg.min_events, cfg.max_events, cfg.p_set, cfg.p_noise, cfg.p_query
    )
    mcfg = cfg.model
    mcfg.vocab_size = task.vocab_size

    model = GPT(mcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    print(
        f"[train] residual={mcfg.residual} qk_norm={mcfg.qk_norm} params={human(model.num_params())} "
        f"device={device} vocab={task.vocab_size}"
    )

    model.train()
    history = []
    for step in range(cfg.max_steps):
        ids, tgt = make_batch(task, cfg.batch_size, mcfg.block_size, rng)
        ids, tgt = ids.to(device), tgt.to(device)

        gamma = None
        if mcfg.residual == "ledger" and cfg.warmup_gamma_steps > 0:
            gamma = max(0.0, 1.0 - step / cfg.warmup_gamma_steps)

        logits, loss = model(ids, tgt, gamma=gamma)
        if cfg.commit_sparsity > 0 and "commit_rate" in model.aux:
            loss = loss + cfg.commit_sparsity * model.aux["commit_rate"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step % cfg.log_every == 0 or step == cfg.max_steps - 1:
            acc = query_accuracy(logits, tgt)
            gtxt = f"{gamma:.2f}" if gamma is not None else "-"
            cpl = model.aux.get("commit_per_layer")
            ctxt = f" | commit {[round(x, 2) for x in cpl]}" if cpl else ""
            print(f"step {step:5d} | loss {loss.item():.4f} | query_acc {acc:.3f} | gamma {gtxt}{ctxt}")
            history.append({"step": step, "loss": loss.item(), "acc": acc, "commit_per_layer": cpl})

    return model, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=None, help="override cfg.max_steps")
    ap.add_argument("--device", default=None, help="override cfg.device")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.device is not None:
        cfg.device = args.device
    train(cfg)


if __name__ == "__main__":
    main()
