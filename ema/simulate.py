"""Closed-loop simulation of a follower and the error metrics.

The model only receives the measured leader speed and the measured initial state;
the measured follower trajectory is used for scoring only.
"""
import numpy as np

from ema import config


class FunctionController:
    """Wrap a function accel(s, v, v_lead) as a stateless controller."""

    def __init__(self, fn):
        self.fn = fn

    def reset(self, s0, v0, v_lead0):
        pass

    def accel(self, s, v, v_lead):
        return self.fn(s, v, v_lead)


class ReplayController:
    """Returns a given acceleration series step by step (e.g. the measured one).

    Replaying the measured acceleration gives the error floor of the simulation.
    """

    def __init__(self, accel_series):
        self.series = np.asarray(accel_series, dtype=float)
        self.k = 0

    def reset(self, s0, v0, v_lead0):
        self.k = 0

    def accel(self, s, v, v_lead):
        a = self.series[self.k]
        self.k += 1
        return a


def rollout(controller, v_lead, s0, v0, dt, collision_gap=0.0, lead_pos=None):
    """Explicit Euler rollout: v(k+1) = max(0, v + a dt).

    Spacing: without lead_pos, s(k+1) = s + (v_lead - v) dt. With lead_pos (leader
    position per step), x_f(k+1) = x_f + v dt with x_f(0) = lead_pos(0) - s0, and
    s(k+1) = lead_pos(k+1) - x_f(k+1).

    Stops at a collision (s <= collision_gap) or a non-finite acceleration (diverged).
    Returns dict(s, v, a, collided, diverged, n_valid); arrays are NaN after a stop.
    """
    v_lead = np.asarray(v_lead, dtype=float)
    n = len(v_lead)
    s, v, a = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    s[0], v[0] = float(s0), float(v0)
    if lead_pos is not None:
        lead_pos = np.asarray(lead_pos, dtype=float)
        x_f = lead_pos[0] - s[0]
    controller.reset(s[0], v[0], v_lead[0])
    for k in range(n - 1):
        acc = float(controller.accel(s[k], v[k], v_lead[k]))
        if not np.isfinite(acc):
            return dict(s=s, v=v, a=a, collided=False, diverged=True, n_valid=k + 1)
        a[k] = acc
        v[k + 1] = max(0.0, v[k] + acc * dt)
        if lead_pos is None:
            s[k + 1] = s[k] + (v_lead[k] - v[k]) * dt
        else:
            x_f += v[k] * dt
            s[k + 1] = lead_pos[k + 1] - x_f
        if s[k + 1] <= collision_gap:
            return dict(s=s, v=v, a=a, collided=True, diverged=False, n_valid=k + 2)
    return dict(s=s, v=v, a=a, collided=False, diverged=False, n_valid=n)


def metrics(sim, s_obs, v_obs):
    """Spacing and speed RMSE over the whole run; NaN if the run collided or diverged."""
    failed = sim["collided"] or sim["diverged"]
    out = dict(collided=bool(sim["collided"]), diverged=bool(sim["diverged"]),
               n_steps=int(len(s_obs)), n_valid=int(sim["n_valid"]))
    if failed:
        out.update(spacing_rmse=np.nan, speed_rmse=np.nan)
    else:
        out.update(spacing_rmse=float(np.sqrt(np.mean((sim["s"] - np.asarray(s_obs)) ** 2))),
                   speed_rmse=float(np.sqrt(np.mean((sim["v"] - np.asarray(v_obs)) ** 2))))
    return out


def leader_position(s_obs, v_obs, dt):
    """Leader position implied by the record: measured spacing + distance driven by the
    measured follower (same Euler rule as the simulation, follower starts at 0)."""
    v_obs = np.asarray(v_obs, dtype=float)
    x_f_obs = np.concatenate([[0.0], np.cumsum(v_obs[:-1] * dt)])
    return np.asarray(s_obs, dtype=float) + x_f_obs


def simulate_run(controller, run_df, cfg=None):
    """Simulate one run (DataFrame with columns v_l, s, v_f, sorted by time) and score it."""
    cfg = cfg or config.load_yaml("evaluation.yaml")
    sc = cfg["simulation"]
    s_obs, v_obs = run_df["s"].to_numpy(float), run_df["v_f"].to_numpy(float)
    if sc.get("spacing_from", "speed_difference") == "leader_position":
        lead_pos = leader_position(s_obs, v_obs, sc["dt_s"])
    elif sc.get("spacing_from", "speed_difference") == "speed_difference":
        lead_pos = None
    else:
        raise ValueError(f"Unknown simulation.spacing_from: {sc['spacing_from']}")
    sim = rollout(controller, run_df["v_l"].to_numpy(float), s_obs[0], v_obs[0],
                  dt=sc["dt_s"], collision_gap=sc["collision_gap_m"], lead_pos=lead_pos)
    return sim, metrics(sim, s_obs, v_obs)
