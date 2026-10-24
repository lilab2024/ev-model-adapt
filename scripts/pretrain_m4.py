"""Pretrain the M4 network on synthetic IDM vehicles. Results go to results/m4_pretrain/.

Usage (from the repository root):
    python scripts/pretrain_m4.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from ema import config, pretrain


def main():
    dt = config.load_yaml("evaluation.yaml")["simulation"]["dt_s"]
    out = config.REPO_ROOT / "results" / "m4_pretrain"
    print(f"torch {torch.__version__}, {torch.get_num_threads()} threads", flush=True)
    summary = pretrain.run(out, dt=dt, log=lambda m: print(m, flush=True))
    info = (out / "run_info.json").read_text(encoding="utf-8")
    print()
    print(f"Saved to {out}")
    print(info)
    with pd.option_context("display.width", 160):
        print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
