import numpy as np
import pandas as pd
import pytest

from ema import calibrate, context, evaluate, methods, protocol, sim_calibrate, simulate
from ema.models import idm

DT = 0.1
P_EV = dict(v0=33.0, T=1.0, s0=2.0, a=1.5, b=2.0, delta=4.0)    # vehicle A
P_ICE = dict(v0=33.0, T=2.2, s0=4.0, a=0.6, b=1.0, delta=4.0)   # vehicle C
CFG_E = {
    "runs": {"qc_pass_only": True, "include_stop_runs": True},
    "task_keys": ["powertrain", "vehicle", "gap"],
    "config_keys": ["vehicle", "gap", "profile", "desired_mph", "v_free_mph", "v_reduced_mph"],
    "levels": ["L2_new_vehicle"],
    "validation": {"fraction_of_configs": 0.25, "seed": 1},
    "context": {"query_fraction": 0.5, "lengths_s": [10, 40], "draws": 2, "seed": 1},
    "simulation": {"dt_s": DT, "collision_gap_m": 0.0, "spacing_from": "leader_position"},
}
CFG_M = {
    "idm": {"params": {"v0": [25.0, 60.0], "T": [0.3, 4.0], "s0": [0.5, 15.0],
                       "a": [0.2, 4.0], "b": [0.2, 6.0], "delta": 4.0},
            "min_gap_m": 0.1},
    "sim_calibration": {"segment_s": 30.0, "segments_per_task": 4, "collision_cost_m": 100.0,
                        "optimizer": {"popsize": 10, "maxiter": 25, "tol": 1e-6,
                                      "polish": True},
                        "seed": 0},
}


def two_vehicle_dataset():
    """EV A and ICE C (two gaps each, 4 configurations x 1 run of 60 s) driven by IDMs
    with different parameters behind a leader with a speed dip."""
    runs, trajs = [], []
    t = np.arange(600) * DT
    for pt, veh, gaps, p in [("EV", "A", ["S", "L"], P_EV), ("ICE", "C", ["min", "max"], P_ICE)]:
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
    runs, traj = two_vehicle_dataset()
    rt = protocol.run_table(runs, CFG_E)
    folds = protocol.make_folds(rt, CFG_E)
    asg = protocol.assign_runs(rt, folds, CFG_E)
    ct = context.context_table(rt, CFG_E)
    return rt, folds, asg, ct, calibrate.forward_difference(traj, DT)


def test_cut_into_segments():
    df = pd.DataFrame({"s": np.arange(25.0), "v_f": 1.0, "v_l": 1.0})
    lens = [len(c) for c in sim_calibrate.cut_into_segments([df, df.iloc[:11]], 10)]
    assert lens == [10, 10, 5, 10]          # the 1-sample tail of the second piece is dropped


def test_population_simulation_equals_single_sets():
    rt, folds, asg, ct, traj_fd = setup()
    seg = sim_calibrate.make_segments(traj_fd, rt, rt.index, CFG_M, DT)
    pop = {k: [P_EV[k], P_ICE[k]] for k in P_EV}
    rmse, failed = sim_calibrate.simulate_idm_population(pop, seg, DT)
    for i, p in enumerate([P_EV, P_ICE]):
        r1, f1 = sim_calibrate.simulate_idm_batch(p, seg, DT)
        assert np.allclose(rmse[i], r1, equal_nan=True) and (failed[i] == f1).all()


def test_context_segments_have_the_requested_length_and_exclude_queries():
    rt, folds, asg, ct, traj_fd = setup()
    runs_by_id = {rid: g.reset_index(drop=True) for rid, g in traj_fd.groupby("run_id")}
    for (task, draw, length), c in ct[ct["role"] == "context"].groupby(["task", "draw", "length_s"]):
        pieces = evaluate.context_data(c, runs_by_id)
        assert sum(len(p) - 1 for p in pieces) * DT == pytest.approx(length, abs=0.11)
        q = ct[(ct["role"] == "query") & (ct["task"] == task) & (ct["draw"] == draw)]
        assert not set(c["run_id"]) & set(q["run_id"])


def test_m1_adapts_to_the_held_out_vehicle_and_reuses_fits():
    methods.ContextRecalibration._cache.clear()
    rt, folds, asg, ct, traj_fd = setup()
    res1, st1 = evaluate.run_evaluation(methods.ContextRecalibration, CFG_M, rt, folds, asg, ct,
                                        traj_fd, CFG_E, verbose=False)
    assert set(res1["length_s"]) == {10.0, 40.0}
    assert {"ctx_idm_T", "ctx_cost"} <= set(res1.columns)
    short_ctx = res1[res1["length_s"] == 10.0]
    long_ctx = res1[res1["length_s"] == 40.0]
    assert not long_ctx["collided"].any()
    # a longer context identifies the vehicle better (data are noise-free IDM trajectories)
    assert long_ctx["spacing_rmse"].mean() < 1.0 < short_ctx["spacing_rmse"].mean()
    a_long = long_ctx[long_ctx["task"].str.startswith("EV")]
    assert a_long["ctx_idm_T"].median() == pytest.approx(P_EV["T"], rel=0.1)
    # M0 trained on the other vehicle is clearly worse on the held-out one
    res0, _ = evaluate.run_evaluation(methods.NoAdaptationSim, CFG_M, rt, folds, asg, ct,
                                      traj_fd, CFG_E, verbose=False)
    assert res0["spacing_rmse"].mean() > 10 * long_ctx["spacing_rmse"].mean()
    # second pass: every fit comes from the cache
    _, st2 = evaluate.run_evaluation(methods.ContextRecalibration, CFG_M, rt, folds, asg, ct,
                                     traj_fd, CFG_E, verbose=False)
    assert sum(s["n_new_fits"] for s in st2) == 0
    assert sum(s["n_new_fits"] for s in st1) == len(
        ct[ct["role"] == "context"].groupby(["task", "draw", "length_s"]))
