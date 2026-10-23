"""Run the evaluation protocol for one method and summarize the results."""
import time

import numpy as np
import pandas as pd

from ema import simulate


def evaluate_fold(method, fold, rt, ct, traj_fd, runs_by_id, train_run_ids, cfg_e):
    """Fit the method on the training runs of one fold, then simulate every query run of
    every held-out task and draw. Returns one row per (task, draw, length, query run)."""
    method.fit(traj_fd, rt, train_run_ids)
    test = ct[ct["task"].isin(fold["test"])]
    rows = []
    for (task, draw), q in test[test["role"] == "query"].groupby(["task", "draw"]):
        if method.uses_context:
            raise NotImplementedError("Methods that use the context are added in a later step")
        settings = [(np.nan, None)]
        for length, context in settings:
            controller = method.controller(context)
            for rid in q["run_id"]:
                _, m = simulate.simulate_run(controller, runs_by_id[rid], cfg_e)
                rows.append(dict(fold_id=fold["fold_id"], level=fold["level"],
                                 method=method.name, task=task, draw=draw,
                                 length_s=length, run_id=rid, **m))
    return pd.DataFrame(rows)


def run_evaluation(method_cls, cfg_m, rt, folds, assignment, ct, traj_fd, cfg_e, verbose=True,
                   on_fold_done=None):
    """Evaluate a method on the given folds. A new method object is created for every fold.
    on_fold_done(fold, results, state) is called after each fold (e.g. to save it).
    Returns (results DataFrame, list of per-fold states); empty if folds is empty."""
    runs_by_id = {rid: g.reset_index(drop=True) for rid, g in traj_fd.groupby("run_id", sort=False)}
    results, states = [], []
    for f in folds:
        t0 = time.time()
        a = assignment[assignment["fold_id"] == f["fold_id"]]
        train_ids = a.loc[a["role"] == "train", "run_id"].tolist()
        method = method_cls(cfg_m, cfg_e)
        res = evaluate_fold(method, f, rt, ct, traj_fd, runs_by_id, train_ids, cfg_e)
        results.append(res)
        states.append(dict(fold_id=f["fold_id"], level=f["level"], **method.state()))
        if on_fold_done is not None:
            on_fold_done(f, res, states[-1])
        if verbose:
            print(f"  {f['fold_id']:24s} {len(res):5d} simulations, "
                  f"{int(res['collided'].sum()):3d} collided, {time.time() - t0:6.1f} s", flush=True)
    return (pd.concat(results, ignore_index=True) if results else pd.DataFrame()), states


def summarize(results, rt):
    """Per-task and per-level summaries, for all runs and for runs without stops.

    Order of averaging: query runs (non-collided runs for RMSE) -> draws -> tasks
    (every task has the same weight)."""
    d = results.join(rt[["is_stop_run"]], on="run_id")
    per_task_parts, per_level_parts = [], []
    for subset, dd in [("all_runs", d), ("no_stop_runs", d[~d["is_stop_run"]])]:
        per_draw = (dd.groupby(["level", "task", "draw", "length_s"], dropna=False)
                      .agg(spacing_rmse=("spacing_rmse", "mean"),
                           speed_rmse=("speed_rmse", "mean"),
                           collision_rate=("collided", "mean"),
                           n_runs=("run_id", "size")))
        per_task = (per_draw.groupby(["level", "task", "length_s"], dropna=False)
                            [["spacing_rmse", "speed_rmse", "collision_rate"]].mean()
                            .assign(subset=subset))
        per_level = (per_task.groupby(["level", "length_s"], dropna=False)
                             .agg(n_tasks=("spacing_rmse", "size"),
                                  spacing_rmse=("spacing_rmse", "mean"),
                                  speed_rmse=("speed_rmse", "mean"),
                                  collision_rate=("collision_rate", "mean"))
                             .assign(subset=subset))
        per_task_parts.append(per_task)
        per_level_parts.append(per_level)
    return pd.concat(per_task_parts).reset_index(), pd.concat(per_level_parts).reset_index()
