"""Synthetic IDM vehicles for pretraining (D017); no real data are used.

Every virtual vehicle has one IDM parameter set, drawn log-uniformly within the bounds
of config/models.yaml (delta fixed), and several runs behind programmatic leader speed
profiles (dip, oscillation, steps; rate-limited). Followers are simulated with the same
Euler rule as the evaluation; records get measurement noise; runs whose true gap leaves
[min_gap_m, max_gap_m] are dropped; noisy records are smoothed (Savitzky-Golay) like
the processed real data. The output has the same format as the real data
(a run table with a task per vehicle and 10 Hz trajectories), so the existing pipeline
(segments, contexts, simulation) can be reused.
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from ema import calibrate, config
from ema.models import idm


def sample_params(rng, spec, n):
    """n IDM parameter sets, log-uniform within the calibrated bounds; fixed values kept."""
    free, bounds, fixed = calibrate.split_spec(spec)
    out = {k: np.full(n, v) for k, v in fixed.items()}
    for name, (lo, hi) in zip(free, bounds):
        out[name] = np.exp(rng.uniform(np.log(lo), np.log(hi), n))
    return out


def leader_speeds(rng, n_runs, n_steps, dt, lc, v_cap):
    """Rate-limited leader speeds (n_runs x n_steps) following random target profiles."""
    t = np.arange(n_steps) * dt
    base = rng.uniform(lc["base_speed_mps"][0], lc["base_speed_mps"][1], n_runs)
    base = np.minimum(base, 0.9 * v_cap)
    kinds = rng.choice(lc["kinds"], n_runs)
    target = np.repeat(base[:, None], n_steps, axis=1)
    for i in range(n_runs):
        if kinds[i] == "dip":
            t0 = rng.uniform(*lc["dip_start_s"])
            if rng.uniform() < lc.get("dip_stop_share", 0.0):
                low = 0.0
            else:
                low = base[i] * (1.0 - rng.uniform(*lc["dip_depth_frac"]))
            ramp = (base[i] - low) / lc["decel_mps2"][0]
            t1 = t0 + ramp + rng.uniform(*lc["hold_s"])
            target[i, (t >= t0) & (t < t1)] = low
        elif kinds[i] == "osc":
            amp = min(rng.uniform(*lc["osc_amplitude_mps"]), base[i])
            period = rng.uniform(*lc["osc_period_s"])
            target[i] = base[i] + amp * np.sin(2 * np.pi * t / period + rng.uniform(0, 2 * np.pi))
        else:  # step
            level, t_next = base[i], rng.uniform(*lc["step_every_s"])
            for k in range(n_steps):
                if t[k] >= t_next:
                    level = float(np.clip(level + rng.choice([-1, 1]) * rng.uniform(*lc["step_size_mps"]),
                                          0.3 * base[i], min(base[i] + 5.0, 0.9 * v_cap[i])))
                    t_next += rng.uniform(*lc["step_every_s"])
                target[i, k] = level
    decel = rng.uniform(*lc["decel_mps2"], n_runs)
    accel = rng.uniform(*lc["accel_mps2"], n_runs)
    v = np.empty((n_runs, n_steps))
    v[:, 0] = target[:, 0]
    for k in range(n_steps - 1):
        v[:, k + 1] = v[:, k] + np.clip(target[:, k] - v[:, k], -decel * dt, accel * dt)
    return np.maximum(v, 0.0), kinds


def equilibrium_gap(v, p):
    """IDM equilibrium gap at speed v (no relative speed)."""
    s_star = p["s0"] + v * p["T"]
    return s_star / np.sqrt(np.maximum(1.0 - (v / p["v0"]) ** p["delta"], 1e-6))


def make_dataset(cfg_pre=None, cfg_models=None, dt=0.1, n_vehicles=None, noise=True):
    """Returns (runs, traj, params): runs = run table (run_id, task, vehicle, kind,
    duration_s), traj = 10 Hz records (run_id, t, v_f, v_l, s), params = DataFrame of the
    IDM parameters per vehicle (index = task)."""
    cfg_pre = cfg_pre or config.load_yaml("pretraining.yaml")
    cfg_models = cfg_models or config.load_yaml("models.yaml")
    sc = cfg_pre["synthetic"]
    rng = np.random.default_rng(int(sc["seed"]))
    n_veh = int(n_vehicles or sc["n_vehicles"])
    n_runs_v = int(sc["runs_per_vehicle"])
    n_steps = int(round(sc["run_s"] / dt)) + 1
    min_gap = cfg_models["idm"]["min_gap_m"]

    pv = sample_params(rng, cfg_models["idm"]["params"], n_veh)
    veh = np.repeat(np.arange(n_veh), n_runs_v)
    p = {k: v[veh] for k, v in pv.items()}
    n = len(veh)
    v_lead, kinds = leader_speeds(rng, n, n_steps, dt, sc["leader"], p["v0"])

    x_lead = np.concatenate([np.zeros((n, 1)), np.cumsum(v_lead[:, :-1] * dt, axis=1)], axis=1)
    v = v_lead[:, 0].copy()
    s = equilibrium_gap(v, p) * rng.uniform(*sc["init_gap_factor"], n)
    x = x_lead[:, 0] - s
    S, V = np.empty((n, n_steps)), np.empty((n, n_steps))
    S[:, 0], V[:, 0] = s, v
    for k in range(n_steps - 1):
        a = idm.accel(S[:, k], V[:, k], v_lead[:, k], p, min_gap)
        x = x + V[:, k] * dt
        V[:, k + 1] = np.maximum(0.0, V[:, k] + a * dt)
        S[:, k + 1] = x_lead[:, k + 1] - x

    ok = (np.isfinite(S).all(axis=1) & (S.min(axis=1) >= sc["reject"]["min_gap_m"])
          & (S.max(axis=1) <= sc["reject"]["max_gap_m"]))
    if noise:
        S_rec = S + rng.normal(0.0, sc["noise"]["spacing_m"], S.shape)
        V_rec = np.maximum(0.0, V + rng.normal(0.0, sc["noise"]["speed_mps"], V.shape))
        VL_rec = np.maximum(0.0, v_lead + rng.normal(0.0, sc["noise"]["speed_mps"], v_lead.shape))
        w = int(round(sc["smoothing"]["window_s"] / dt)) | 1          # odd window length
        order = int(sc["smoothing"]["polyorder"])
        S_rec, V_rec, VL_rec = (savgol_filter(x, w, order, axis=1) for x in (S_rec, V_rec, VL_rec))
        V_rec, VL_rec = np.maximum(V_rec, 0.0), np.maximum(VL_rec, 0.0)
    else:
        S_rec, V_rec, VL_rec = S, V, v_lead

    t = np.arange(n_steps) * dt
    keep = np.where(ok)[0]
    run_ids = [f"syn{veh[i]:05d}_r{i % n_runs_v}" for i in keep]
    tasks = [f"SYN | {veh[i]:05d}" for i in keep]
    runs = pd.DataFrame({"run_id": run_ids, "task": tasks, "vehicle": [f"{veh[i]:05d}" for i in keep],
                         "kind": kinds[keep], "duration_s": float(t[-1])}).set_index("run_id")
    traj = pd.DataFrame({
        "run_id": np.repeat(run_ids, n_steps), "t": np.tile(t, len(keep)),
        "v_f": V_rec[keep].ravel(), "v_l": VL_rec[keep].ravel(), "s": S_rec[keep].ravel()})
    params = pd.DataFrame({k: v for k, v in pv.items()},
                          index=pd.Index([f"SYN | {i:05d}" for i in range(n_veh)], name="task"))
    params = params.loc[sorted(set(tasks))]
    return runs, traj, params
