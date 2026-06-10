"""Modal GPU entrypoint for the headline mechanistic LM run (thin wrapper over ledger.train_lm).

    modal run modal_lm.py --config configs/lm_fineweb.yaml                 # single config
    modal run modal_lm.py::sweep --config configs/lm_fineweb.yaml          # vanilla/ledger/suppress in parallel

Trains real LMs on FineWeb-Edu and measures the per-sublayer massive-activation profile
(norm, max/median, Sun-threshold counts). See docs/research_plan.md §7.3 and docs/CLAUDE.md.
No training logic lives here — it calls train_lm(cfg).
"""
from __future__ import annotations

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.2", "numpy", "pyyaml", "datasets", "tiktoken")
    .add_local_python_source("ledger")
)
app = modal.App("ledger-residuals-lm")
data_vol = modal.Volume.from_name("ledger-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("ledger-ckpts", create_if_missing=True)


def _make_cfg(config_yaml: str, override: dict | None = None, max_steps=None):
    import yaml
    from ledger.config import LMConfig, ModelConfig, _filter

    raw = yaml.safe_load(config_yaml) or {}
    model_raw = raw.pop("model", {}) or {}
    if override:
        model_raw.update(override.get("model", {}))
        raw.update({k: v for k, v in override.items() if k != "model"})
    cfg = LMConfig(model=ModelConfig(**_filter(ModelConfig, model_raw)), **_filter(LMConfig, raw))
    cfg.out_dir = "/ckpts/" + cfg.out_dir
    if max_steps is not None:
        cfg.max_steps = max_steps
    return cfg


@app.function(image=image, gpu="A100", timeout=6 * 3600, volumes={"/data": data_vol, "/ckpts": ckpt_vol})
def run_lm(config_yaml: str, override: dict | None = None, max_steps=None):
    from ledger.train_lm import train_lm

    cfg = _make_cfg(config_yaml, override, max_steps)
    _model, history = train_lm(cfg)
    ckpt_vol.commit()
    last = history[-1] if history else {}
    return {"residual": cfg.model.residual, "qk_norm": cfg.model.qk_norm, "n_layer": cfg.model.n_layer,
            "val_loss": last.get("val_loss")}


@app.local_entrypoint()
def main(config: str, max_steps: int = None, gpu: str = "A100"):
    with open(config) as fh:
        config_yaml = fh.read()
    print(run_lm.remote(config_yaml, max_steps=max_steps))


@app.local_entrypoint()
def sweep(config: str, max_steps: int = None):
    """Run the three canonical configs in parallel (matched everything but the residual op)."""
    with open(config) as fh:
        config_yaml = fh.read()
    overrides = [
        {"out_dir": "outputs/lm/vanilla", "model": {"residual": "vanilla"}},
        {"out_dir": "outputs/lm/suppress", "model": {"residual": "vanilla", "qk_norm": True}},
        {"out_dir": "outputs/lm/ledger", "model": {"residual": "ledger", "gamma": 0.0}},
    ]
    calls = [run_lm.spawn(config_yaml, override=ov, max_steps=max_steps) for ov in overrides]
    for c in calls:
        print(c.get())
