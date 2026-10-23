import numpy as np
import pandas as pd
import pytest

from ema import calibrate, sim_calibrate, simulate
from ema.models import idm

P = dict(v0=33.0, T=1.5, s0=2.0, a=1.0, b=1.5, delta=4.0)
DT = 0.1
CFG_M = {
    "idm": {"params": {"v0": [25.0, 60.0], "T": [0.3, 4.0], "s0": [0.5, 15.0],
                       "a": [0.2, 4.0], "b": [0.2, 6.0], "delta": 4.0},
            "min_gap_m": 0.1},
    "sim_calibration": {"segment_s": 30.0, "segments_per_task": 4, "collision_cost_m": 100.0,
                        "optimizer": {"popsize": 10, "maxiter": 25, "tol": 1e-6,
                                      "polish": True},
                        "seed": 0},
}
SIM_CFG = {"simulation": {"dt_s": DT, "collision_gap_m": 0.0, "spacing_from": "leader_position"}}


def idm_runs(params=P, n_runs=4, n=400):
    """Runs driven by the IDM behind a leader with a speed dip; one task per two runs."""
    t = np.arange(n) * DT
    trajs, tasks = [], {}
    for k in range(n_runs):
        rid = f"r{k}"
        v_lead = 20.0 - 6.0 * (1 + 0.2 * k) * np.exp(-((t - 15.0) / 5.0) ** 2)
        sim = simulate.rollout(idm.IDMController(params), v_lead, s0=40.0, v0=20.0, dt=DT)
        trajs.append(pd.DataFrame({"run_id": rid, "t": t, "v_f": sim["v"],
                                   "v_l": v_lead, "s": sim["s"]}))
        tasks[rid] = "A" if k < n_runs // 2 else "B"
    traj_fd = calibrate.forward_difference(pd.concat(trajs, ignore_index=True), DT)
    rt = pd.DataFrame({"task": pd.Series(tasks)}).rename_axis("run_id")
    return traj_fd, rt


def test_segments_are_task_balanced_and_within_runs():
    traj_fd, rt = idm_runs()
    seg = sim_calibrate.make_segments(traj_fd, rt, rt.index, CFG_M, DT)
    tasks = [lab[0] for lab in seg["labels"]]
    assert tasks.count("A") == tasks.count("B") == 4
    assert (seg["length"] == 300).all()


def test_batch_simulation_matches_the_scalar_simulation():
    traj_fd, rt = idm_runs()
    seg = sim_calibrate.make_segments(traj_fd, rt, rt.index, CFG_M, DT)
    p2 = dict(v0=40.0, T=1.2, s0=3.0, a=1.4, b=2.0, delta=4.0)
    rmse, failed = sim_calibrate.simulate_idm_batch(p2, seg, DT)
    assert not failed.any()
    for i, (task, rid, start) in enumerate(seg["labels"]):
        g = traj_fd[traj_fd["run_id"] == rid].iloc[start:start + seg["length"][i]]
        _, m = simulate.simulate_run(idm.IDMController(p2), g, SIM_CFG)
        assert rmse[i] == pytest.approx(m["spacing_rmse"], rel=1e-9, abs=1e-9)


def test_batch_simulation_flags_collisions():
    n = 50
    s_obs = np.linspace(5.0, -1.0, n)            # gap closes to zero while the follower stands
    seg = dict(v_lead=np.zeros((1, n)), s_obs=s_obs[None, :],
               lead_pos=simulate.leader_position(s_obs, np.zeros(n), DT)[None, :],
               length=np.array([n]), s0=np.array([5.0]), v0=np.array([0.0]))
    rmse, failed = sim_calibrate.simulate_idm_batch(P, seg, DT)
    assert failed[0] and np.isnan(rmse[0])


def test_calibration_recovers_the_generating_parameters():
    traj_fd, rt = idm_runs()
    seg = sim_calibrate.make_segments(traj_fd, rt, rt.index, CFG_M, DT)
    params, cost, nfev = sim_calibrate.calibrate_idm_sim(seg, CFG_M, DT)
    assert cost < 0.01
    for k in ["T", "s0", "a", "b"]:
        assert params[k] == pytest.approx(P[k], rel=0.01)


def test_method_uses_only_given_runs_and_reports_its_state():
    from ema import methods
    traj_fd, rt = idm_runs()
    cfg = {**CFG_M, "sim_calibration": {**CFG_M["sim_calibration"],
                                        "optimizer": {"popsize": 5, "maxiter": 2, "tol": 1e-6,
                                                      "polish": False}}}
    m = methods.NoAdaptationSim(cfg, SIM_CFG)
    m.fit(traj_fd, rt, ["r0", "r2"])
    st = m.state()
    assert m.train_run_ids == ["r0", "r2"] and st["n_segments"] == 8
    assert {"train_cost", "n_evaluations", "idm_T"} <= set(st)


def test_method_requires_leader_position_spacing():
    from ema import methods
    traj_fd, rt = idm_runs()
    bad = {"simulation": {**SIM_CFG["simulation"], "spacing_from": "speed_difference"}}
    with pytest.raises(ValueError):
        methods.NoAdaptationSim(CFG_M, bad).fit(traj_fd, rt, ["r0"])
