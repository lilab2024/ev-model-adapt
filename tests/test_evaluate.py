import numpy as np
import pandas as pd

from ema import calibrate, context, evaluate, methods, protocol, simulate
from ema.models import idm

P = dict(v0=33.0, T=1.5, s0=2.0, a=1.0, b=1.5, delta=4.0)
DT = 0.1
CFG_E = {
    "runs": {"qc_pass_only": True, "include_stop_runs": True},
    "task_keys": ["powertrain", "vehicle", "gap"],
    "config_keys": ["vehicle", "gap", "profile", "desired_mph", "v_free_mph", "v_reduced_mph"],
    "levels": ["L2_new_vehicle"],
    "validation": {"fraction_of_configs": 0.25, "seed": 1},
    "context": {"query_fraction": 0.5, "lengths_s": [10, 30], "draws": 2, "seed": 1},
    "simulation": {"dt_s": DT, "collision_gap_m": 0.0, "spacing_from": "leader_position"},
}
CFG_M = {
    "idm": {"params": {"v0": [25.0, 60.0], "T": [0.3, 4.0], "s0": [0.5, 15.0],
                       "a": [0.2, 4.0], "b": [0.2, 6.0], "delta": 4.0},
            "min_gap_m": 0.1},
    "calibration": {"target": "forward_difference", "samples_per_task": 400,
                    "n_starts": 2, "seed": 0},
}


def synthetic_dataset():
    """Two vehicles (EV A, ICE C), two gaps each, 4 configurations x 1 run of 60 s,
    all driven by the IDM with parameters P behind a leader with a speed dip."""
    runs, trajs = [], []
    t = np.arange(600) * DT
    for pt, veh, gaps in [("EV", "A", ["S", "L"]), ("ICE", "C", ["min", "max"])]:
        for gap in gaps:
            for k in range(4):
                rid = f"{pt}_{veh}_{gap}_{k}"
                v_lead = 20.0 - 6.0 * (1 + 0.2 * k) * np.exp(-((t - 25.0) / 6.0) ** 2)
                sim = simulate.rollout(idm.IDMController(P), v_lead, s0=40.0, v0=20.0, dt=DT)
                trajs.append(pd.DataFrame({"run_id": rid, "t": t, "v_f": sim["v"],
                                           "v_l": v_lead, "s": sim["s"]}))
                runs.append(dict(run_id=rid, powertrain=pt, vehicle=veh, gap=gap, profile="dip",
                                 desired_mph=0.0, v_free_mph=45.0, v_reduced_mph=20.0 + 5 * k,
                                 qc_pass=True, exclude=False, duration_s=float(t[-1])))
    return pd.DataFrame(runs), pd.concat(trajs, ignore_index=True)


def run_m0():
    runs, traj = synthetic_dataset()
    rt = protocol.run_table(runs, CFG_E)
    folds = protocol.make_folds(rt, CFG_E)
    asg = protocol.assign_runs(rt, folds, CFG_E)
    ct = context.context_table(rt, CFG_E)
    traj_fd = calibrate.forward_difference(traj, DT)
    res, states = evaluate.run_evaluation(methods.NoAdaptation, CFG_M, rt, folds, asg, ct,
                                          traj_fd, CFG_E, verbose=False)
    return rt, folds, asg, res, states


def test_m0_scores_almost_perfectly_on_idm_generated_data():
    rt, folds, asg, res, states = run_m0()
    # 2 folds x 2 test tasks x 2 draws x 2 query runs
    assert len(res) == 16
    assert not res["collided"].any()
    assert res["length_s"].isna().all()
    assert res["spacing_rmse"].max() < 0.01


def test_summary_counts_tasks_and_collisions():
    rt, folds, asg, res, states = run_m0()
    per_task, per_level = evaluate.summarize(res, rt)
    lvl = per_level[per_level["subset"] == "all_runs"]
    assert lvl["n_tasks"].tolist() == [4]
    assert (lvl["collision_rate"] == 0).all()


def test_m0_learns_only_from_training_runs_of_the_fold():
    runs, traj = synthetic_dataset()
    rt = protocol.run_table(runs, CFG_E)
    folds = protocol.make_folds(rt, CFG_E)
    asg = protocol.assign_runs(rt, folds, CFG_E)
    traj_fd = calibrate.forward_difference(traj, DT)
    for f in folds:
        a = asg[asg["fold_id"] == f["fold_id"]]
        m = methods.NoAdaptation(CFG_M)
        m.fit(traj_fd, rt, a.loc[a["role"] == "train", "run_id"].tolist())
        used = set(m.train_run_ids)
        assert used and not used & set(a.loc[a["role"].isin(["test", "val"]), "run_id"])


def test_callback_after_each_fold_and_empty_fold_list():
    runs, traj = synthetic_dataset()
    rt = protocol.run_table(runs, CFG_E)
    folds = protocol.make_folds(rt, CFG_E)
    asg = protocol.assign_runs(rt, folds, CFG_E)
    ct = context.context_table(rt, CFG_E)
    traj_fd = calibrate.forward_difference(traj, DT)
    seen = []
    res, states = evaluate.run_evaluation(
        methods.NoAdaptation, CFG_M, rt, folds[:1], asg, ct, traj_fd, CFG_E, verbose=False,
        on_fold_done=lambda f, r, s: seen.append((f["fold_id"], len(r), s["method"])))
    assert seen == [(folds[0]["fold_id"], len(res), "no_adaptation")]
    empty, no_states = evaluate.run_evaluation(methods.NoAdaptation, CFG_M, rt, [], asg, ct,
                                               traj_fd, CFG_E, verbose=False)
    assert empty.empty and no_states == []
