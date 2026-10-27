"""Physics-model floor comparison (D028): IDM, OVRV and IDM with actuation lag, on
training and validation runs only.

Usage (from the repository root):
    python scripts/physics_floor.py

For every task and model, the model is calibrated in closed loop on the task's validation
runs (in-sample floor) and on its training runs (cross-configuration floor), and both are
scored on the validation runs (as diagnostic C, D026). Each task is saved to
results/diagnostics/physics_floor/tasks/ when finished and reused by a later call from the
same clean commit and config files. At the end: run, task and summary tables, and the
rule of D028 (a model is clearly better than the IDM only if its task-averaged
cross-configuration floor is at least min_improvement_m below the IDM's for EV and ICE).
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ema import calibrate, config, data, diagnostics, physics_models, protocol

CONFIG_FILES = ["data.yaml", "evaluation.yaml", "models.yaml", "physics_models.yaml"]


def git_state():
    root = config.REPO_ROOT
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=root).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                           cwd=root).stdout.strip() != ""
    return head, dirty


def config_md5():
    h = hashlib.md5()
    for name in CONFIG_FILES:
        h.update((config.CONFIG_DIR / name).read_bytes())
    return h.hexdigest()


def safe(task):
    return task.replace(" | ", "__").replace(" ", "_").replace("/", "-")


def decide(summary, min_improvement):
    """Rule of D028 on the task-averaged cross-configuration floors."""
    cc = summary[summary["floor"] == "cross_config"].set_index(["model", "powertrain"])["task_mean_rmse"]
    out = {}
    for model in physics_models.MODELS:
        if model == "idm":
            continue
        gains = {pt: float(cc[("idm", pt)] - cc[(model, pt)]) for pt in ("EV", "ICE")}
        out[model] = dict(gain_m=gains, clearly_better=all(g >= min_improvement for g in gains.values()))
    return out


def main():
    cfg_e, cfg_m = config.load_yaml("evaluation.yaml"), config.load_yaml("models.yaml")
    cfg_p = config.load_yaml("physics_models.yaml")
    head, dirty = git_state()
    cfg_hash = config_md5()
    rt = protocol.run_table(data.load_runs(), cfg_e)
    traj_fd = calibrate.forward_difference(data.load_trajectories(), cfg_e["simulation"]["dt_s"])
    runs_by_id = {rid: g.reset_index(drop=True) for rid, g in traj_fd.groupby("run_id", sort=False)}
    out = config.REPO_ROOT / "results" / "diagnostics" / "physics_floor"
    (out / "tasks").mkdir(parents=True, exist_ok=True)
    tasks = sorted(rt["task"].unique())
    print(f"physics floor: {len(tasks)} tasks, models {list(physics_models.MODELS)}, "
          f"git {head[:7]}{' DIRTY' if dirty else ''}", flush=True)

    parts = []
    for task in tasks:
        f_csv, f_json = out / "tasks" / f"{safe(task)}.csv", out / "tasks" / f"{safe(task)}.json"
        if f_json.exists() and not dirty:
            meta = json.loads(f_json.read_text(encoding="utf-8"))
            if meta["git_commit"] == head and not meta["git_dirty"] and meta["config_md5"] == cfg_hash:
                parts.append(pd.read_csv(f_csv) if meta["n_rows"] else pd.DataFrame())
                print(f"  {task}: reused", flush=True)
                continue
        t0 = time.time()
        rows = []
        for model in physics_models.MODELS:
            rows += diagnostics.model_floor(task, model, traj_fd, rt, runs_by_id, cfg_m, cfg_p, cfg_e)
        df = pd.DataFrame(rows)
        df.to_csv(f_csv, index=False)
        f_json.write_text(json.dumps(dict(task=task, git_commit=head, git_dirty=dirty,
                                          config_md5=cfg_hash, n_rows=len(df))), encoding="utf-8")
        parts.append(df)
        msg = "no validation runs" if df.empty else "; ".join(
            f"{m} {g[g['floor'] == 'in_sample']['spacing_rmse'].mean():.2f}/"
            f"{g[g['floor'] == 'cross_config']['spacing_rmse'].mean():.2f}"
            for m, g in df.groupby("model", sort=False))
        print(f"  {task}: {msg} m ({time.time() - t0:.0f} s)", flush=True)

    runs = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    runs["powertrain"] = runs["task"].str.split(" | ", regex=False).str[0]
    runs.to_csv(out / "physics_floor_runs.csv", index=False)
    per_task = (runs.groupby(["powertrain", "task", "model", "floor"])
                .agg(n_runs=("run_id", "size"), spacing_rmse_mean=("spacing_rmse", "mean"),
                     n_collided=("collided", "sum"))
                .reset_index())
    per_task.to_csv(out / "physics_floor_tasks.csv", index=False)
    summary = (per_task.groupby(["model", "powertrain", "floor"])
               .agg(n_tasks=("task", "size"), task_mean_rmse=("spacing_rmse_mean", "mean"),
                    task_median_rmse=("spacing_rmse_mean", "median"),
                    n_collided=("n_collided", "sum"))
               .reset_index())
    summary.to_csv(out / "physics_floor_summary.csv", index=False)
    decision = decide(summary, float(cfg_p["rule"]["min_improvement_m"]))
    (out / "run_info.json").write_text(json.dumps(dict(
        git_commit=head, git_dirty=dirty, config_md5=cfg_hash, n_tasks=len(tasks),
        decision=decision), indent=2), encoding="utf-8")
    print()
    print(f"Saved to {out} (git {head[:7]}{' DIRTY' if dirty else ''})")
    print(summary.round(3).to_string(index=False))
    for model, d in decision.items():
        print(f"rule (D028): {model}: cross-config gain vs IDM EV {d['gain_m']['EV']:+.2f} m, "
              f"ICE {d['gain_m']['ICE']:+.2f} m -> "
              f"{'clearly better' if d['clearly_better'] else 'not clearly better'}")


if __name__ == "__main__":
    main()
