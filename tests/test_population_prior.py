import numpy as np
import pandas as pd
import pytest

from ema import calibrate, context, evaluate, methods, protocol, sim_calibrate, simulate
from ema.models import idm

DT = 0.1
BASE = dict(v0=33.0, T=1.5, s0=2.5, a=1.0, b=1.5, delta=4.0)
VEHICLES = [("EV", "A", ["S", "L"], 0.8), ("ICE", "C", ["min", "max"], 1.0),
            ("ICE", "D", ["min", "max"], 1.2)]            # factor on T and 1/factor on a
CFG_E = {
    "runs": {"qc_pass_only": True, "include_stop_runs": True},
    "task_keys": ["powertrain", "vehicle", "gap"],
    "config_keys": ["vehicle", "gap", "profile", "desired_mph", "v_free_mph", "v_reduced_mph"],
    "levels": ["L2_new_vehicle"],
    "validation": {"fraction_of_configs": 0.25, "seed": 1},
    "context": {"query_fraction": 0.5, "lengths_s": [10, 40], "draws": 1, "seed": 1},
    "simulation": {"dt_s": DT, "collision_gap_m": 0.0, "spacing_from": "leader_position"},
}
CFG_M = {
    "idm": {"params": {"v0": [25.0, 60.0], "T": [0.3, 4.0], "s0": [0.5, 15.0],
                       "a": [0.2, 4.0], "b": [0.2, 6.0], "delta": 4.0},
            "min_gap_m": 0.1},
    "sim_calibration": {"segment_s": 30.0, "segments_per_task": 3, "collision_cost_m": 100.0,
                        "optimizer": {"popsize": 8, "maxiter": 15, "tol": 1e-6, "polish": True},
                        "seed": 0},
    "population_prior": {"lambdas_m": [0.3, 3.0], "pseudo_tasks": 2, "pseudo_context_s": [10],
                         "min_log_std": 0.05, "min_tasks_for_selection": 3, "seed": 0},
}


def dataset():
    runs, trajs = [], []
    t = np.arange(600) * DT
    for pt, veh, gaps, f in VEHICLES:
        p = {**BASE, "T": BASE["T"] * f, "a": BASE["a"] / f}
        for gap in gaps:
            for k in range(4):
                rid = f"{pt}_{veh}_{gap}_{k}"
                v_lead = 20.0 - 6.0 * (1 + 0.2 * k) * np.exp(-((t - 25.0) / 6.0) ** 2)
                sim = simulate.rollout(idm.IDMController(p), v_lead, s0=40.0, v0=20.0, dt=DT)
                trajs.append(pd.DataFrame({"run_id": rid, "t": t, "v_f": sim["v"],
                                           "v_l": v_lead, "s": sim["s"]}))
                runs.append(dict(run_id=rid, powertrain=pt, vehicle=veh, gap=gap, profile="dip",
                                 desired_mph=0.0, v_free_mph=45.0, v_reduced_mph=20.0 + 5 * k,
                                 qc_pass=True, exclude=False, duration_s=float(t[-1])))
    return pd.DataFrame(runs), pd.concat(trajs, ignore_index=True)


def setup():
    runs, traj = dataset()
    rt = protocol.run_table(runs, CFG_E)
    folds = protocol.make_folds(rt, CFG_E)
    asg = protocol.assign_runs(rt, folds, CFG_E)
    ct = context.context_table(rt, CFG_E)
    return rt, folds, asg, ct, calibrate.forward_difference(traj, DT)


def test_make_prior_uses_log_mean_and_floored_sd():
    sets = [dict(BASE), {**BASE, "T": BASE["T"] * np.e}]
    pr = sim_calibrate.make_prior(sets, ["T", "a"], lam=2.0, min_log_std=0.05)
    assert pr["mu"][0] == pytest.approx(np.log(BASE["T"]) + 0.5)
    assert pr["sd"][0] == pytest.approx(np.std([0.0, 1.0], ddof=1))
    assert pr["sd"][1] == pytest.approx(0.05) and pr["lam"] == 2.0


def test_zero_weight_prior_equals_no_prior():
    rt, folds, asg, ct, traj_fd = setup()
    seg = sim_calibrate.make_segments(traj_fd, rt, rt.index[:4], CFG_M, DT)
    free, _, fixed = calibrate.split_spec(CFG_M["idm"]["params"])
    pr = sim_calibrate.make_prior([BASE], free, lam=0.0)
    x = np.array([[40.0, 35.0], [1.2, 2.0], [3.0, 1.0], [1.4, 0.9], [2.0, 1.0]])
    a = sim_calibrate.objective(x, free, fixed, seg, DT, 0.0, 0.1, 100.0)
    b = sim_calibrate.objective(x, free, fixed, seg, DT, 0.0, 0.1, 100.0, pr)
    assert np.allclose(a, b)


def test_strong_prior_pulls_the_fit_to_the_prior_mean():
    rt, folds, asg, ct, traj_fd = setup()
    seg = sim_calibrate.make_segments(traj_fd, rt, rt.index[:2], CFG_M, DT)
    free = calibrate.split_spec(CFG_M["idm"]["params"])[0]
    target = dict(v0=45.0, T=2.5, s0=5.0, a=0.6, b=3.0, delta=4.0)
    pr = sim_calibrate.make_prior([target], free, lam=1e6, min_log_std=0.05)
    params, _, _ = sim_calibrate.calibrate_idm_sim(seg, CFG_M, DT, prior=pr)
    for k in free:
        assert params[k] == pytest.approx(target[k], rel=0.02)


def test_m2_uses_only_training_tasks_and_chooses_lambda_from_the_grid():
    methods.PopulationPrior._task_fits.clear()
    rt, folds, asg, ct, traj_fd = setup()
    f = folds[0]
    res, st = evaluate.run_evaluation(methods.PopulationPrior, CFG_M, rt, [f], asg, ct,
                                      traj_fd, CFG_E, verbose=False)
    s = st[0]
    assert s["prior_lambda"] in CFG_M["population_prior"]["lambdas_m"]
    assert s["n_train_tasks"] == len(f["train"])
    assert set(s["pseudo_tasks"].split("; ")) <= set(f["train"])
    assert {"ctx_lambda", "ctx_idm_T"} <= set(res.columns)
    assert set(res["length_s"]) == {10.0, 40.0}
    assert not res["collided"].any()
