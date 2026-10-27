"""Diagnostics on training and validation runs only (no test results; D013).

idm_floor (diagnostic C, D026): for one task, the closed-loop spacing error that the IDM
reaches on the task's validation runs (a) when calibrated on those same runs (in-sample
floor) and (b) when calibrated on the task's training runs, as in M2's task fit
(cross-configuration floor).
"""
import numpy as np

from ema import protocol, sim_calibrate, simulate
from ema.models import idm


def task_split(rt, task, cfg_eval):
    """(training run ids, validation run ids) of a task; validation runs are those of its
    validation configurations (identical in every fold)."""
    cfgs = protocol.validation_configs(rt, task, cfg_eval)
    sub = rt[rt["task"] == task]
    val = sorted(sub.index[sub["config_id"].isin(cfgs)])
    train = sorted(sub.index[~sub["config_id"].isin(cfgs)])
    return train, val


def calibrate_on(traj_fd, rt, run_ids, cfg_models, cfg_eval):
    """IDM calibrated in closed loop on segments of the given runs (as M2's task fit)."""
    sim = cfg_eval["simulation"]
    seg = sim_calibrate.make_segments(traj_fd, rt, run_ids, cfg_models, sim["dt_s"])
    params, _, _ = sim_calibrate.calibrate_idm_sim(seg, cfg_models, sim["dt_s"],
                                                   sim["collision_gap_m"])
    return params


def idm_floor(task, traj_fd, rt, runs_by_id, cfg_models, cfg_eval):
    """Rows (one per validation run and floor) with the closed-loop errors of the
    in-sample and cross-configuration IDM fits; empty if the task has no validation or no
    training runs."""
    train, val = task_split(rt, task, cfg_eval)
    if not train or not val:
        return []
    fits = {"in_sample": calibrate_on(traj_fd, rt, val, cfg_models, cfg_eval),
            "cross_config": calibrate_on(traj_fd, rt, train, cfg_models, cfg_eval)}
    rows = []
    for floor, params in fits.items():
        ctrl = idm.IDMController(params, cfg_models["idm"]["min_gap_m"])
        for r in val:
            _, m = simulate.simulate_run(ctrl, runs_by_id[r], cfg_eval)
            rows.append(dict(task=task, run_id=r, floor=floor, n_train_runs=len(train),
                             n_val_runs=len(val), spacing_rmse=float(m["spacing_rmse"]),
                             speed_rmse=float(m.get("speed_rmse", np.nan)),
                             collided=bool(m["collided"]), diverged=bool(m["diverged"]),
                             **{f"idm_{k}": float(v) for k, v in params.items()}))
    return rows
