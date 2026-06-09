"""Modal entrypoint for GPU runs (thin wrapper over ledger.train — NO logic here).

All experiments are designed to run on Modal (the user has no local GPU). Local CPU
smoke tests use `python -m ledger.train` directly; this file is the GPU path.

    modal run modal_app.py --config configs/toy.yaml
    modal run modal_app.py --config configs/ledger.yaml --gpu A100

Data and checkpoints live in Modal Volumes; the wandb key (when used) comes from a
Modal Secret. See docs/CLAUDE.md (Modal rules) and docs/TASKS.md Phase 6.

NOTE: this is the Phase-6 scaffold. It is not exercised by the local CPU smoke tests
(modal is an optional dependency). Verify end-to-end on Modal before the first real run.
"""
from __future__ import annotations

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.2", "numpy", "pyyaml", "wandb", "datasets")
    .add_local_python_source("ledger")  # mounts src/ledger into the container
)

app = modal.App("ledger-residuals")

# Persistent storage: datasets (download once) and run artifacts (checkpoints/logs).
data_vol = modal.Volume.from_name("ledger-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("ledger-ckpts", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",  # override per call; use "H100" for the ~1B tier
    timeout=60 * 60 * 6,
    volumes={"/data": data_vol, "/ckpts": ckpt_vol},
    # secrets=[modal.Secret.from_name("wandb")],  # enable when logging to wandb
)
def run_training(config_yaml: str, max_steps: int | None = None):
    import yaml
    from ledger.config import ModelConfig, TrainConfig, _filter
    from ledger.train import train

    raw = yaml.safe_load(config_yaml) or {}
    model_raw = raw.pop("model", {}) or {}
    cfg = TrainConfig(model=ModelConfig(**_filter(ModelConfig, model_raw)), **_filter(TrainConfig, raw))
    cfg.out_dir = "/ckpts/" + cfg.out_dir
    if max_steps is not None:
        cfg.max_steps = max_steps

    _model, history = train(cfg)
    ckpt_vol.commit()
    return history[-1] if history else {}


@app.local_entrypoint()
def main(config: str, max_steps: int | None = None, gpu: str = "A100"):
    # `gpu` is accepted for documentation/forwarding; set the decorator default per tier.
    with open(config) as fh:
        config_yaml = fh.read()
    result = run_training.remote(config_yaml, max_steps=max_steps)
    print("final:", result)
