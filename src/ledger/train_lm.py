"""Language-model training + the headline mechanistic measurement.

    python -m ledger.train_lm --config configs/lm_smoke.yaml          # CPU/MPS smoke (pseudo-text)
    modal run modal_lm.py --config configs/lm_fineweb.yaml            # GPU on Modal (FineWeb-Edu)

Trains a next-token LM and, at intervals, measures the residual-stream massive-activation
profile (norm, max/median ratio, outlier counts) per sublayer for vanilla / ledger / suppress.
This is where the thesis is actually testable: massive activations are a scale + real-language
phenomenon that the synthetic toy could not exercise.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch

from .config import LMConfig, load_lm_config
from .data.text import build_tokenizer, load_token_ids, TokenDataset
from .model import GPT
from .utils import pick_device, set_seed, human


@torch.no_grad()
def evaluate(model, data: TokenDataset, cfg, device, gen, n_batches=20):
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = data.batch(cfg.batch_size, device, gen)
        _, loss = model(x, y)
        losses.append(loss.item())
    return sum(losses) / len(losses)


@torch.no_grad()
def measure_activations(model, data: TokenDataset, cfg, device, gen):
    """Per-sublayer massive-activation profile on a fresh batch (decode regime, gamma=model.gamma)."""
    model.eval()
    x, _ = data.batch(cfg.batch_size, device, gen)
    model(x, record_trace=True)
    return model.aux["trace"]


def train_lm(cfg: LMConfig):
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    gen = torch.Generator().manual_seed(cfg.seed)

    tok = build_tokenizer(cfg.tokenizer)
    ids = load_token_ids(cfg.data, tok, cfg.max_tokens + cfg.val_tokens, seed=cfg.seed)
    train_ids, val_ids = ids[: -cfg.val_tokens], ids[-cfg.val_tokens:]
    train_data = TokenDataset(train_ids, cfg.model.block_size)
    val_data = TokenDataset(val_ids, cfg.model.block_size)

    mcfg = cfg.model
    mcfg.vocab_size = tok.vocab_size
    model = GPT(mcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def lr_at(step):
        if step < cfg.warmup_steps:
            return cfg.lr * (step + 1) / cfg.warmup_steps
        t = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        return 0.1 * cfg.lr + 0.9 * cfg.lr * 0.5 * (1 + math.cos(math.pi * t))

    print(f"[lm] residual={mcfg.residual} qk_norm={mcfg.qk_norm} layers={mcfg.n_layer} dim={mcfg.dim} "
          f"params={human(model.num_params())} tok={cfg.tokenizer}({tok.vocab_size}) data={cfg.data} device={device}")

    history = []
    model.train()
    for step in range(cfg.max_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = train_data.batch(cfg.batch_size, device, gen)
        _, loss = model(x, y)
        if cfg.commit_sparsity > 0 and "commit_rate" in model.aux:
            loss = loss + cfg.commit_sparsity * model.aux["commit_rate"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step % cfg.log_every == 0 or step == cfg.max_steps - 1:
            print(f"step {step:5d} | train_loss {loss.item():.4f} | lr {lr_at(step):.2e}")
        if step > 0 and (step % cfg.eval_every == 0 or step == cfg.max_steps - 1):
            vloss = evaluate(model, val_data, cfg, device, gen)
            trace = measure_activations(model, val_data, cfg, device, gen)
            maxmed = [round(t["primary_maxmed"], 1) for t in trace]
            print(f"  >> step {step} val_loss {vloss:.4f} | val_ppl {math.exp(vloss):.2f} | "
                  f"max/med by layer {maxmed}")
            history.append({"step": step, "val_loss": vloss, "trace": trace})
            model.train()

    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, f"lm_{mcfg.residual}{'_qk' if mcfg.qk_norm else ''}_d{mcfg.n_layer}.json"), "w") as fh:
        json.dump({"config": mcfg.__dict__, "history": history}, fh, indent=2)
    return model, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = load_lm_config(args.config)
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.device is not None:
        cfg.device = args.device
    train_lm(cfg)


if __name__ == "__main__":
    main()
