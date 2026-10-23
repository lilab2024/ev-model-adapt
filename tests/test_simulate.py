import numpy as np
import pytest

from ema import simulate
from ema.simulate import FunctionController, ReplayController, rollout, metrics


def test_equilibrium_stays_constant():
    out = rollout(FunctionController(lambda s, v, vl: 0.0), np.full(100, 20.0),
                  s0=30.0, v0=20.0, dt=0.1)
    assert not out["collided"]
    assert np.allclose(out["s"], 30.0) and np.allclose(out["v"], 20.0)


def test_single_euler_step():
    out = rollout(FunctionController(lambda s, v, vl: 1.0), np.array([10.0, 10.0]),
                  s0=20.0, v0=8.0, dt=0.1)
    assert out["v"][1] == pytest.approx(8.1)
    assert out["s"][1] == pytest.approx(20.2)


def test_collision_stops_the_simulation():
    v_lead = np.r_[np.full(50, 20.0), np.zeros(200)]      # leader stops abruptly
    out = rollout(FunctionController(lambda s, v, vl: 0.0), v_lead, s0=10.0, v0=20.0, dt=0.1)
    assert out["collided"] and out["n_valid"] < len(v_lead)
    assert np.isnan(out["s"][-1])


def test_speed_never_negative():
    out = rollout(FunctionController(lambda s, v, vl: -10.0), np.full(50, 5.0),
                  s0=50.0, v0=5.0, dt=0.1)
    assert (out["v"] >= 0).all()


def test_nonfinite_acceleration_is_a_failure():
    out = rollout(FunctionController(lambda s, v, vl: np.nan), np.full(10, 5.0), 20.0, 5.0, 0.1)
    assert out["diverged"]
    m = metrics(out, np.full(10, 20.0), np.full(10, 5.0))
    assert np.isnan(m["spacing_rmse"]) and np.isnan(m["speed_rmse"])


def test_metrics_values():
    sim = dict(s=np.array([1.0, 2.0]), v=np.array([1.0, 1.0]),
               collided=False, diverged=False, n_valid=2)
    m = metrics(sim, np.array([1.0, 4.0]), np.array([1.0, 2.0]))
    assert m["spacing_rmse"] == pytest.approx(np.sqrt(2.0))
    assert m["speed_rmse"] == pytest.approx(np.sqrt(0.5))


def test_replay_reproduces_a_trajectory_built_with_the_same_rule():
    rng = np.random.default_rng(0)
    n, dt = 200, 0.1
    a = rng.normal(0.0, 0.3, n)
    v_lead = 20.0 + np.cumsum(rng.normal(0.0, 0.05, n))
    v, s = np.empty(n), np.empty(n)
    v[0], s[0] = 20.0, 30.0
    for k in range(n - 1):
        v[k + 1] = max(0.0, v[k] + a[k] * dt)
        s[k + 1] = s[k] + (v_lead[k] - v[k]) * dt
    out = rollout(ReplayController(a), v_lead, s[0], v[0], dt)
    assert np.allclose(out["s"], s) and np.allclose(out["v"], v)


LEADER_CFG = {"simulation": {"dt_s": 0.1, "collision_gap_m": 0.0,
                             "spacing_from": "leader_position"}}


def test_leader_position_mode_reproduces_inconsistent_recorded_spacing():
    """If the follower speed is reproduced exactly, the simulated spacing equals the
    recorded spacing even when spacing and speeds are mutually inconsistent."""
    import pandas as pd
    rng = np.random.default_rng(1)
    n, dt = 300, 0.1
    a = rng.normal(0.0, 0.2, n)
    v = 20.0 + np.concatenate([[0.0], np.cumsum(a[:-1] * dt)])   # no clipping needed
    s = 30.0 + np.cumsum(rng.normal(0.0, 0.05, n))               # unrelated to the speeds
    run_df = pd.DataFrame({"v_l": v + 0.3, "s": s, "v_f": v})
    sim, m = simulate.simulate_run(ReplayController(a), run_df, LEADER_CFG)
    assert np.allclose(sim["s"], s) and m["spacing_rmse"] < 1e-9


def test_leader_position_mode_detects_collision():
    lead_pos = np.r_[np.linspace(10.0, 30.0, 11), np.full(40, 30.0)]   # leader stops at 30 m
    out = rollout(FunctionController(lambda s, v, vl: 0.0), np.zeros(51), s0=10.0, v0=20.0,
                  dt=0.1, lead_pos=lead_pos)
    assert out["collided"]
