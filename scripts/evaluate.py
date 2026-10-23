"""Run the evaluation protocol for one method. Results go to results/<method>/ (not in git).

Usage (from the repository root):
    python scripts/evaluate.py --method no_adaptation [--levels L1_new_setting ...]
"""
import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from ema import calibrate, config, context, data, evaluate, methods, protocol


def git_state():
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=config.REPO_ROOT).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                           cwd=config.REPO_ROOT).stdout.strip() != ""
    return head, dirty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(methods.REGISTRY))
    ap.add_argument("--levels", nargs="*", default=None)
    args = ap.parse_args()

    t_start = time.time()
    cfg_e = config.load_yaml("evaluation.yaml")
    cfg_m = config.load_yaml("models.yaml")
    rt = protocol.run_table(data.load_runs(), cfg_e)
    folds = protocol.make_folds(rt, cfg_e)
    if args.levels:
        folds = [f for f in folds if f["level"] in args.levels]
    assignment = protocol.assign_runs(rt, folds, cfg_e)
    ct = context.context_table(rt, cfg_e)
    traj_fd = calibrate.forward_difference(data.load_trajectories(), cfg_e["simulation"]["dt_s"])

    print(f"Method {args.method}: {len(folds)} folds, {len(rt)} runs, {rt['task'].nunique()} tasks")
    results, states = evaluate.run_evaluation(methods.REGISTRY[args.method], cfg_m, rt, folds,
                                              assignment, ct, traj_fd, cfg_e)
    per_task, per_level = evaluate.summarize(results, rt)

    out = config.REPO_ROOT / "results" / args.method
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "runs.csv", index=False)
    pd.DataFrame(states).to_csv(out / "fold_states.csv", index=False)
    per_task.to_csv(out / "summary_per_task.csv", index=False)
    per_level.to_csv(out / "summary_per_level.csv", index=False)
    head, dirty = git_state()
    info = dict(method=args.method, levels=args.levels, n_folds=len(folds),
                n_simulations=int(len(results)), git_commit=head, git_dirty=dirty,
                data_md5=config.load_yaml("data.yaml")["processed"]["md5"]["trajectories"],
                finished=datetime.datetime.now().isoformat(timespec="seconds"),
                runtime_s=round(time.time() - t_start, 1))
    (out / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"\nSaved to {out} (git {head[:7]}{' DIRTY' if dirty else ''}), "
          f"{info['runtime_s']} s")
    with pd.option_context("display.width", 160):
        print(per_level.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
