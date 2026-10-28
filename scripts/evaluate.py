"""Run the evaluation protocol for one method. Results go to results/<method>/ (not in git).

Usage (from the repository root):
    python scripts/evaluate.py --method no_adaptation [--levels L1_new_setting ...] [--fresh]

Every fold is saved to results/<method>/folds/ as soon as it is finished. A new call
reuses saved folds only if they were produced by the same git commit (without
uncommitted changes) and the same config files; --fresh recomputes all folds.

Parallel runs: start n processes with --shard k/n --no-summary (k = 0..n-1); each runs
only its share of the folds and writes no summary. A final call without these options
then reuses all saved folds and writes the summary.
"""
import argparse
import datetime
import hashlib
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


def config_md5():
    """md5 over the config files that determine the results (m4.yaml holds the md5 of the
    M4 network weights, which is checked when they are loaded)."""
    h = hashlib.md5()
    for name in ["data.yaml", "evaluation.yaml", "models.yaml", "pretraining.yaml", "m4.yaml",
                 "residual.yaml", "features_gbm.yaml", "transformer.yaml"]:
        h.update((config.CONFIG_DIR / name).read_bytes())
    return h.hexdigest()


def _json_default(o):
    return o.item() if hasattr(o, "item") else str(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(methods.REGISTRY))
    ap.add_argument("--levels", nargs="*", default=None)
    ap.add_argument("--fresh", action="store_true", help="ignore saved folds")
    ap.add_argument("--shard", default=None, help="k/n: run only every n-th fold, from k")
    ap.add_argument("--no-summary", action="store_true", help="save folds only (for --shard)")
    args = ap.parse_args()
    if args.shard and not args.no_summary:
        ap.error("--shard needs --no-summary (the summary is written by a final call)")

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

    out = config.REPO_ROOT / "results" / args.method
    fold_dir = out / "folds"
    fold_dir.mkdir(parents=True, exist_ok=True)
    head, dirty = git_state()
    cfg_hash = config_md5()
    if dirty:
        print("WARNING: uncommitted changes; these folds will not be reused later")

    mine = evaluate.shard(folds, args.shard) if args.shard else folds
    done = set()
    if not args.fresh:
        for f in mine:
            meta_path = fold_dir / f"{f['fold_id']}.json"
            if meta_path.exists() and (fold_dir / f"{f['fold_id']}.csv").exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if (meta["git_commit"] == head and not meta["git_dirty"]
                        and meta["config_md5"] == cfg_hash):
                    done.add(f["fold_id"])
    todo = [f for f in mine if f["fold_id"] not in done]
    print(f"Method {args.method}: {len(mine)} folds{f' (shard {args.shard})' if args.shard else ''} "
          f"({len(done)} reused, {len(todo)} to run), "
          f"{len(rt)} runs, {rt['task'].nunique()} tasks", flush=True)

    def save_fold(fold, res, state):
        res.to_csv(fold_dir / f"{fold['fold_id']}.csv", index=False)
        meta = dict(fold_id=fold["fold_id"], git_commit=head, git_dirty=dirty,
                    config_md5=cfg_hash, state=state,
                    finished=datetime.datetime.now().isoformat(timespec="seconds"))
        (fold_dir / f"{fold['fold_id']}.json").write_text(
            json.dumps(meta, indent=2, default=_json_default), encoding="utf-8")

    evaluate.run_evaluation(methods.REGISTRY[args.method], cfg_m, rt, todo,
                            assignment, ct, traj_fd, cfg_e, on_fold_done=save_fold,
                            artifact_dir=out)
    if args.no_summary:
        print(f"folds saved to {fold_dir} (git {head[:7]}{' DIRTY' if dirty else ''}); "
              "no summary written", flush=True)
        return

    # assemble all requested folds from the saved files
    results = pd.concat([pd.read_csv(fold_dir / f"{f['fold_id']}.csv") for f in folds],
                        ignore_index=True)
    states = [json.loads((fold_dir / f"{f['fold_id']}.json").read_text(encoding="utf-8"))["state"]
              for f in folds]
    per_task, per_level = evaluate.summarize(results, rt)
    results.to_csv(out / "runs.csv", index=False)
    pd.DataFrame(states).to_csv(out / "fold_states.csv", index=False)
    per_task.to_csv(out / "summary_per_task.csv", index=False)
    per_level.to_csv(out / "summary_per_level.csv", index=False)
    info = dict(method=args.method, levels=args.levels, n_folds=len(folds),
                n_folds_reused=len(done), n_simulations=int(len(results)),
                git_commit=head, git_dirty=dirty, config_md5=cfg_hash,
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
