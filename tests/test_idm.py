import numpy as np
import pandas as pd
import pytest

from ema import calibrate
from ema.models import idm

P = dict(v0=33.0, T=1.5, s0=2.0, a=1.0, b=1.5, delta=4.0)


def test_free_road_start_uses_max_acceleration():
    assert idm.accel(1e6, 0.0, 0.0, P) == pytest.approx(P["a"])


def test_equilibrium_acceleration():
    v = 10.0
    s_eq = P["s0"] + v * P["T"]
    assert idm.accel(s_eq, v, v, P) == pytest.approx(-P["a"] * (v / P["v0"]) ** 4)


def test_brakes_harder_when_closer():
    assert idm.accel(5.0, 15.0, 10.0, P) < idm.accel(20.0, 15.0, 10.0, P) < 0


def test_controller_matches_vectorized_formula():
    ctrl = idm.IDMController(P)
    for s, v, vl in [(30.0, 20.0, 18.0), (3.0, 0.0, 0.0), (0.05, 5.0, 0.0)]:
        assert ctrl.accel(s, v, vl) == pytest.approx(float(idm.accel(s, v, vl, P)))


def test_forward_difference_per_run():
    traj = pd.DataFrame({"run_id": ["a", "a", "a", "b", "b"], "t": [0.0, 0.1, 0.2, 0.0, 0.1],
                         "v_f": [1.0, 2.0, 4.0, 10.0, 9.0]})
    fd = calibrate.forward_difference(traj, 0.1).set_index(["run_id", "t"])["a_fd"]
    assert fd[("a", 0.0)] == pytest.approx(10.0) and fd[("a", 0.1)] == pytest.approx(20.0)
    assert np.isnan(fd[("a", 0.2)]) and np.isnan(fd[("b", 0.1)])
    assert fd[("b", 0.0)] == pytest.approx(-10.0)


def test_sample_per_task_is_balanced():
    traj = pd.DataFrame({"run_id": ["r1"] * 50 + ["r2"] * 5, "t": list(range(50)) + list(range(5)),
                         "v_f": 1.0, "s": 1.0, "v_l": 1.0, "a_fd": 0.0})
    rt = pd.DataFrame({"task": ["A", "B"]}, index=pd.Index(["r1", "r2"], name="run_id"))
    smp = calibrate.sample_per_task(traj, rt, ["r1", "r2"], n_per_task=10, seed=1)
    assert smp["task"].value_counts().to_dict() == {"A": 10, "B": 5}


def test_calibration_recovers_known_parameters():
    rng = np.random.default_rng(0)
    n = 4000
    s = rng.uniform(5.0, 80.0, n)
    v = rng.uniform(0.0, 30.0, n)
    vl = v + rng.uniform(-3.0, 3.0, n)
    samples = pd.DataFrame({"s": s, "v_f": v, "v_l": vl, "a_fd": idm.accel(s, v, vl, P)})
    cfg = {"idm": {"params": {"v0": [25.0, 60.0], "T": [0.3, 4.0], "s0": [0.5, 15.0],
                              "a": [0.2, 4.0], "b": [0.2, 6.0], "delta": 4.0},
                   "min_gap_m": 0.1},
           "calibration": {"n_starts": 3, "seed": 0}}
    params, rmse = calibrate.calibrate_idm(samples, cfg)
    assert rmse < 1e-6
    for k in ["v0", "T", "s0", "a", "b"]:
        assert params[k] == pytest.approx(P[k], rel=1e-3)
