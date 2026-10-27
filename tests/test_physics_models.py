import numpy as np
import pandas as pd
import pytest

from ema import calibrate, diagnostics, physics_models, protocol, sim_calibrate, simulate
from ema.models import idm
from test_population_prior import CFG_E, CFG_M, DT, setup

CFG_P = {"ovrv": {"params": {"k1": [0.001, 1.0], "k2": [0.01, 2.0], "eta": [0.5, 15.0],
                             "tau": [0.3, 4.0]}, "accel_limits": [-8.0, 4.0]},
         "idm_lag": {"tau_a": [0.05, 2.0]}, "rule": {"min_improvement_m": 0.3}}
P_IDM = dict(v0=33.0, T=1.5, s0=2.5, a=1.0, b=1.5, delta=4.0)
P_OVRV = dict(k1=0.08, k2=0.6, eta=3.0, tau=1.6)


def segments():
    rt, folds, asg, ct, traj_fd = setup()
    return rt, traj_fd, sim_calibrate.make_segments(traj_fd, rt, rt.index, CFG_M, DT)


def test_the_config_file_has_the_agreed_bounds():
    from ema import config
    c = config.load_yaml("physics_models.yaml")
    assert c["ovrv"]["params"] == CFG_P["ovrv"]["params"]
    assert c["ovrv"]["accel_limits"] == [-8.0, 4.0] and c["idm_lag"]["tau_a"] == [0.05, 2.0]
    assert c["rule"]["min_improvement_m"] == 0.3


def test_generic_idm_simulation_equals_the_idm_calibration_code():
    rt, traj_fd, seg = segments()
    P = {k: [v, v * 1.1] if k == "T" else v for k, v in P_IDM.items()}
    a = physics_models.simulate_population("idm", P, seg, DT)
    b = sim_calibrate.simulate_idm_population(P, seg, DT)
    assert np.array_equal(a[1], b[1]) and np.allclose(a[0], b[0], equal_nan=True, rtol=0, atol=0)
    p1, c1 = physics_models.calibrate("idm", seg, CFG_M, CFG_P, DT)
    p2, c2, _ = sim_calibrate.calibrate_idm_sim(seg, CFG_M, DT)
    assert p1 == p2 and c1 == c2


def test_idm_lag_without_lag_equals_the_idm():
    rt, traj_fd, seg = segments()
    a = physics_models.simulate_population("idm_lag", {**P_IDM, "tau_a": 1e-9}, seg, DT)
    b = physics_models.simulate_population("idm", P_IDM, seg, DT)
    assert np.allclose(a[0], b[0], equal_nan=True)
    slow = physics_models.simulate_population("idm_lag", {**P_IDM, "tau_a": 1.5}, seg, DT)
    assert not np.allclose(slow[0], b[0], equal_nan=True)


@pytest.mark.parametrize("model,params", [("ovrv", P_OVRV), ("idm_lag", {**P_IDM, "tau_a": 0.7})])
def test_batch_simulation_equals_the_scalar_controller(model, params):
    rt, traj_fd, seg = segments()
    rmse, failed = physics_models.simulate_population(model, {k: [v] for k, v in params.items()},
                                                      seg, DT)
    for i in range(3):
        n = int(seg["length"][i])
        ctrl = physics_models.controller(model, params, CFG_M, CFG_P, DT)
        sim = simulate.rollout(ctrl, seg["v_lead"][i, :n], seg["s0"][i], seg["v0"][i], DT,
                               lead_pos=seg["lead_pos"][i, :n])
        err = np.sqrt(np.sum((sim["s"][1:n] - seg["s_obs"][i, 1:n]) ** 2) / n)
        assert err == pytest.approx(rmse[0, i], rel=1e-9) and not failed[0, i]


def test_ovrv_acceleration_is_clipped():
    c = physics_models.OVRVController(P_OVRV, (-8.0, 4.0))
    assert c.accel(500.0, 0.0, 30.0) == 4.0 and c.accel(0.5, 30.0, 0.0) == -8.0
    a = physics_models.ovrv_accel(np.array([500.0, 0.5]), np.array([0.0, 30.0]),
                                  np.array([30.0, 0.0]), P_OVRV, (-8.0, 4.0))
    assert list(a) == [4.0, -8.0]


def test_ovrv_floor_is_small_on_ovrv_generated_data():
    rt, folds, asg, ct, traj_fd = setup()
    runs_by_id = {}
    parts = []
    for rid, g in traj_fd.groupby("run_id"):          # replace the follower by an OVRV vehicle
        g = g.reset_index(drop=True)
        sim = simulate.rollout(physics_models.OVRVController(P_OVRV), g["v_l"].to_numpy(),
                               40.0, 20.0, DT)
        g = g.assign(s=sim["s"], v_f=sim["v"])
        parts.append(g)
    traj = calibrate.forward_difference(pd.concat(parts, ignore_index=True)[
        ["run_id", "t", "v_f", "v_l", "s"]], DT)
    runs_by_id = {rid: g.reset_index(drop=True) for rid, g in traj.groupby("run_id")}
    task = sorted(rt["task"].unique())[0]
    rows = pd.DataFrame(diagnostics.model_floor(task, "ovrv", traj, rt, runs_by_id, CFG_M,
                                                CFG_P, CFG_E))
    assert set(rows["floor"]) == {"in_sample", "cross_config"} and not rows["collided"].any()
    assert rows["spacing_rmse"].max() < 1.0
