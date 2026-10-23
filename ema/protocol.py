"""Evaluation protocol: runs, tasks, folds and train/val/test assignment.

Everything here is fixed by config/evaluation.yaml and does not depend on any model.
"""
import hashlib

import numpy as np
import pandas as pd

from ema import config


def run_table(runs, cfg=None):
    """Runs used for evaluation, indexed by run_id, with task, config_id and is_stop_run."""
    cfg = cfg or config.load_yaml("evaluation.yaml")
    r = runs.copy()
    r = r[~r["exclude"].astype(bool)]
    if cfg["runs"]["qc_pass_only"]:
        r = r[r["qc_pass"].eq(True)]
    r["is_stop_run"] = (r["powertrain"] == "EV") & (r["v_reduced_mph"] == 0)
    if not cfg["runs"]["include_stop_runs"]:
        r = r[~r["is_stop_run"]]
    r["task"] = r[cfg["task_keys"]].astype(str).agg(" | ".join, axis=1)
    r["config_id"] = r[cfg["config_keys"]].astype(str).agg("|".join, axis=1)
    return r.set_index("run_id").sort_index()


def make_folds(rt, cfg=None):
    """List of folds: dict(fold_id, level, test, train, test_label); test/train are task lists."""
    cfg = cfg or config.load_yaml("evaluation.yaml")
    tasks = rt.groupby("task")[["vehicle", "powertrain"]].first().sort_index()
    all_tasks = list(tasks.index)
    ev = [t for t in all_tasks if tasks.loc[t, "powertrain"] == "EV"]
    ice = [t for t in all_tasks if tasks.loc[t, "powertrain"] == "ICE"]

    def of_vehicle(veh, pool):
        return [t for t in pool if tasks.loc[t, "vehicle"] == veh]

    candidates = []
    for t in all_tasks:
        if len(of_vehicle(tasks.loc[t, "vehicle"], all_tasks)) >= 2:
            candidates.append(("L1_new_setting", [t], [x for x in all_tasks if x != t]))
    for veh in sorted(tasks["vehicle"].unique()):
        test = of_vehicle(veh, all_tasks)
        candidates.append(("L2_new_vehicle", test, [x for x in all_tasks if x not in test]))
    candidates.append(("L3_cross_powertrain", ev, ice))
    for veh in sorted(tasks.loc[ev, "vehicle"].unique()):
        test = of_vehicle(veh, ev)
        candidates.append(("L2_no_ice", test, [x for x in ev if x not in test]))

    folds, counter = [], {}
    for level, test, train in candidates:
        if level not in cfg["levels"]:
            continue
        counter[level] = counter.get(level, 0) + 1
        folds.append(dict(fold_id=f"{level}_{counter[level]:02d}", level=level,
                          test=test, train=train, test_label="; ".join(test)))
    return folds


def _task_seed(seed, task):
    """Seed that depends only on the global seed and the task name (stable across folds)."""
    return [int(seed), int(hashlib.md5(task.encode("utf-8")).hexdigest()[:8], 16)]


def validation_configs(rt, task, cfg=None):
    """Configurations of a task reserved for validation (identical in every fold)."""
    cfg = cfg or config.load_yaml("evaluation.yaml")
    configs = sorted(rt.loc[rt["task"] == task, "config_id"].unique())
    if len(configs) < 2:
        return set()
    n = max(1, int(round(cfg["validation"]["fraction_of_configs"] * len(configs))))
    rng = np.random.default_rng(_task_seed(cfg["validation"]["seed"], task))
    return set(rng.choice(configs, size=n, replace=False))


def assign_runs(rt, folds, cfg=None):
    """Long table: fold_id, level, role (train / val / test), task, config_id, run_id."""
    cfg = cfg or config.load_yaml("evaluation.yaml")
    val_cfg = {t: validation_configs(rt, t, cfg) for t in rt["task"].unique()}
    rows = []
    for f in folds:
        for base_role, tasks in (("train", f["train"]), ("test", f["test"])):
            sub = rt[rt["task"].isin(tasks)]
            for run_id, task, config_id in zip(sub.index, sub["task"], sub["config_id"]):
                role = "val" if base_role == "train" and config_id in val_cfg[task] else base_role
                rows.append(dict(fold_id=f["fold_id"], level=f["level"], role=role,
                                 task=task, config_id=config_id, run_id=run_id))
    return pd.DataFrame(rows)
