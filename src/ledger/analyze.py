"""Analyse a sweep result JSON (outputs/modal_results/<tag>_sweep.json) into the
massive-activation comparison figure + summary.

    python -m ledger.analyze --results outputs/modal_results/lm125_sweep.json

Compares the DECODED channel of each config at matched val loss: vanilla `h`, suppress `h`,
ledger `D` (scratch) and ledger `C` (the decoded commitment channel). The headline metric is
`feat_maxmed` — the fixed-dimension signature that defines Sun-style massive activations —
plus kurtosis and raw absmax (all independent of the norm-growth confound).
"""
from __future__ import annotations

import argparse
import json

METRICS = ["feat_maxmed", "kurtosis", "absmax", "maxmed", "n_out50"]


def _col(trace, key):
    return [t.get(key, float("nan")) for t in trace]


def analyze(path: str, out: str):
    R = json.load(open(path))
    V, S, L = R.get("vanilla"), R.get("suppress"), R.get("ledger")

    print("val_loss:", {k: round(R[k].get("val_loss", float("nan")), 4) for k in R})
    print("\nPEAK value by config (decoded channel in **bold** = vanilla h / suppress h / ledger C):")
    for m in METRICS:
        def pk(d, key):
            vals = [v for v in _col(d["trace"], key) if v == v] if d else []
            return round(max(vals), 2) if vals else None
        print(f"  {m:12s}: vanilla_h={pk(V,'primary_'+m)}  suppress_h={pk(S,'primary_'+m)}  "
              f"ledger_D={pk(L,'primary_'+m)}  ledger_C={pk(L,'secondary_'+m)}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("(matplotlib unavailable; printed summary only)")
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, m in zip(axes.flat, METRICS):
        if V: ax.plot(_col(V["trace"], "primary_" + m), marker=".", label="vanilla h")
        if S: ax.plot(_col(S["trace"], "primary_" + m), marker=".", label="suppress h (QK-norm)")
        if L:
            ax.plot(_col(L["trace"], "primary_" + m), marker=".", label="ledger D (scratch)")
            ax.plot(_col(L["trace"], "secondary_" + m), marker="x", ls="--", lw=2, label="ledger C (DECODED)")
        ax.set(title=m, xlabel="sublayer")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes.flat[-1].axis("off")
    fig.suptitle("Massive-activation profile at matched val loss (GPT-2 BPE, ~160M, FineWeb)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nsaved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default="outputs/modal_results/ma_profile_125m.png")
    a = ap.parse_args()
    analyze(a.results, a.out)


if __name__ == "__main__":
    main()
