"""One-step least-squares calibration of physics models on pooled training samples."""
import hashlib

import numpy as np
import pandas as pd
from scipy import optimize

from ema import config
from ema.models import idm


def _seed(seed, *parts):
    digest = hashlib.md5("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:8]
    return [int(seed), int(digest, 16)]


def forward_difference(traj, dt):
    """Copy of traj sorted by run and time, with a_fd = (v_f[k+1] - v_f[k]) / dt per run.
    The last sample of every run has a_fd = NaN."""
    out = traj.sort_values(["run_id", "t"]).copy()
    out["a_fd"] = -out.groupby("run_id", sort=False)["v_f"].diff(-1) / dt
    return out


def sample_per_task(traj_fd, rt, run_ids, n_per_task, seed):
    """Up to n_per_task samples (with a_fd) from the given runs of every task."""
    d = traj_fd[traj_fd["run_id"].isin(set(run_ids))].dropna(subset=["a_fd"])
    d = d.join(rt["task"], on="run_id")
    parts = []
    for task, g in d.groupby("task", sort=True):
        n = min(int(n_per_task), len(g))
        rng = np.random.default_rng(_seed(seed, task))
        parts.append(g.iloc[np.sort(rng.choice(len(g), size=n, replace=False))])
    return pd.concat(parts, ignore_index=True)


def split_spec(spec):
    """{name: [low, high]} -> calibrated; {name: value} -> fixed."""
    free, bounds, fixed = [], [], {}
    for name, value in spec.items():
        if isinstance(value, (list, tuple)):
            free.append(name)
            bounds.append((float(value[0]), float(value[1])))
        else:
            fixed[name] = float(value)
    return free, bounds, fixed


def calibrate_idm(samples, cfg_models=None):
    """Least squares of IDM acceleration against a_fd. Returns (params, one-step RMSE)."""
    cfg_models = cfg_models or config.load_yaml("models.yaml")
    m, c = cfg_models["idm"], cfg_models["calibration"]
    free, bounds, fixed = split_spec(m["params"])
    lo, hi = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
    s, v, vl, y = (samples[k].to_numpy(float) for k in ["s", "v_f", "v_l", "a_fd"])

    def residual(x):
        return idm.accel(s, v, vl, {**fixed, **dict(zip(free, x))}, m["min_gap_m"]) - y

    rng = np.random.default_rng(int(c["seed"]))
    starts = [(lo + hi) / 2] + [rng.uniform(lo, hi) for _ in range(int(c["n_starts"]) - 1)]
    best = None
    for x0 in starts:
        res = optimize.least_squares(residual, x0, bounds=(lo, hi), method="trf",
                                     x_scale="jac", max_nfev=2000)
        if best is None or res.cost < best.cost:
            best = res
    params = {**fixed, **dict(zip(free, map(float, best.x)))}
    rmse = float(np.sqrt(np.mean(residual(best.x) ** 2)))
    return params, rmse
