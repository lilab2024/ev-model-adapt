"""Pretrain the M4 network on synthetic IDM vehicles. Results go to results/m4_pretrain/.

Usage (from the repository root):
    python scripts/pretrain_m4.py [--resume results/m4_pretrain/net.pt] [--out m4_pretrain_r2]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from ema import config, pretrain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", default=None, help="start from these saved weights")
    ap.add_argument("--out", default="m4_pretrain", help="folder name under results/")
    ap.add_argument("--network", default=None,
                    help="config file whose 'network' replaces that of pretraining.yaml "
                         "(e.g. transformer.yaml, D032)")
    args = ap.parse_args()
    cfg_pre = config.load_yaml("pretraining.yaml")
    if args.network:
        cfg_pre = {**cfg_pre, "network": config.load_yaml(args.network)["network"]}
    print("network:", cfg_pre["network"], flush=True)
    dt = config.load_yaml("evaluation.yaml")["simulation"]["dt_s"]
    out = config.REPO_ROOT / "results" / args.out
    resume = config.REPO_ROOT / args.resume if args.resume else None
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    print(f"torch {torch.__version__}, {torch.get_num_threads()} threads, GPU: {gpu}", flush=True)
    summary = pretrain.run(out, cfg_pre=cfg_pre, dt=dt, log=lambda m: print(m, flush=True),
                           resume_from=resume)
    info = (out / "run_info.json").read_text(encoding="utf-8")
    print()
    print(f"Saved to {out}")
    print(info)
    with pd.option_context("display.width", 160):
        print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
