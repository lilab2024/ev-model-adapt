"""Simulation-based (closed-loop) calibration of the IDM (D012).

Training runs are cut into task-balanced segments; a parameter set is scored by
simulating all segments at once (same Euler rule and leader-position spacing as
ema.simulate) and averaging the spacing RMSE per segment. Differential evolution
searches the parameter bounds of config/models.yaml; the whole population of candidate
parameter sets is simulated at once (vectorized), which gives the same results as
evaluating the candidates one by one, only faster.
"""
import numpy as np
from scipy import optimize

from ema import config, simulate
from ema.calibrate import _seed, split_spec
from ema.models import idm


def make_segments(traj_fd, rt, run_ids, cfg_models=None, dt=0.1):
    """Task-balanced segments from the given runs.

    Every task contributes segments_per_task segments; each segment is a random run of
    the task and a random start; runs shorter than segment_s are used whole.
    Returns a dict of padded arrays (n_segments x max_len) plus lengths and labels.
    """
    cfg_models = cfg_models or config.load_yaml("models.yaml")
    sc = cfg_models["sim_calibration"]
    seg_len = int(round(sc["segment_s"] / dt))
    d = traj_fd[traj_fd["run_id"].isin(set(run_ids))]
    by_run = {rid: g for rid, g in d.groupby("run_id", sort=False)}
    task_of = rt.loc[list(by_run), "task"]

    pieces, labels = [], []
    for task in sorted(task_of.unique()):
        rids = sorted(task_of.index[task_of == task])
        rng = np.random.default_rng(_seed(sc["seed"], task))
        for _ in range(int(sc["segments_per_task"])):
            rid = rids[int(rng.integers(len(rids)))]
            g = by_run[rid]
            n = min(seg_len, len(g))
            start = int(rng.integers(0, len(g) - n + 1))
            pieces.append(g.iloc[start:start + n])
            labels.append((task, rid, start))

    return pack_segments(pieces, dt, labels)


def cut_into_segments(pieces, seg_len):
    """Split each DataFrame into consecutive chunks of at most seg_len samples
    (chunks with fewer than 2 samples are dropped)."""
    out = []
    for p in pieces:
        for start in range(0, len(p), int(seg_len)):
            chunk = p.iloc[start:start + int(seg_len)]
            if len(chunk) >= 2:
                out.append(chunk)
    return out


def pack_segments(pieces, dt, labels=None):
    """Padded arrays (n_segments x max_len) for a list of DataFrames with columns s, v_f, v_l."""
    lengths = np.array([len(p) for p in pieces])
    T = int(lengths.max())
    shape = (len(pieces), T)
    v_lead, s_obs, lead_pos = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    for i, p in enumerate(pieces):
        n = len(p)
        s_i, v_i = p["s"].to_numpy(float), p["v_f"].to_numpy(float)
        v_lead[i, :n] = p["v_l"].to_numpy(float)
        s_obs[i, :n] = s_i
        lead_pos[i, :n] = simulate.leader_position(s_i, v_i, dt)
        v_lead[i, n:], s_obs[i, n:], lead_pos[i, n:] = v_lead[i, n - 1], s_i[-1], lead_pos[i, n - 1]
    return dict(v_lead=v_lead, s_obs=s_obs, lead_pos=lead_pos, length=lengths,
                s0=s_obs[:, 0].copy(), v0=np.array([p["v_f"].iloc[0] for p in pieces], float),
                labels=labels if labels is not None else list(range(len(pieces))))


def simulate_idm_population(P, seg, dt, collision_gap=0.0, min_gap=0.1):
    """Simulate all segments for S parameter sets at once.

    P maps parameter names to arrays of length S (or scalars shared by all sets).
    Returns (spacing RMSE, failed flag), both of shape (S, n_segments). A segment fails if
    it collides (simulated gap <= collision_gap) or the acceleration is not finite.
    """
    S = max(int(np.size(v)) for v in P.values())
    p = {k: (np.asarray(v, dtype=float).reshape(-1, 1) if np.size(v) > 1
             else float(np.asarray(v, dtype=float).ravel()[0])) for k, v in P.items()}
    n, T = seg["v_lead"].shape
    s = np.broadcast_to(seg["s0"].astype(float), (S, n)).copy()
    v = np.broadcast_to(seg["v0"].astype(float), (S, n)).copy()
    x = seg["lead_pos"][:, 0] - s
    failed = np.zeros((S, n), dtype=bool)
    err_sum = np.zeros((S, n))
    length = seg["length"]
    with np.errstate(all="ignore"):
        for k in range(T - 1):
            active = (~failed) & (k + 1 < length)
            if not active.any():
                break
            acc = idm.accel(s, v, seg["v_lead"][:, k], p, min_gap)
            bad = active & ~np.isfinite(acc)
            v_new = np.maximum(0.0, v + acc * dt)
            x_new = x + v * dt
            s_new = seg["lead_pos"][:, k + 1] - x_new
            hit = active & ~bad & (s_new <= collision_gap)
            ok = active & ~bad & ~hit
            failed |= bad | hit
            err_sum = np.where(ok, err_sum + (s_new - seg["s_obs"][:, k + 1]) ** 2, err_sum)
            v = np.where(ok, v_new, v)
            x = np.where(ok, x_new, x)
            s = np.where(ok, s_new, s)
    rmse = np.sqrt(err_sum / length)
    return np.where(failed, np.nan, rmse), failed


def simulate_idm_batch(p, seg, dt, collision_gap=0.0, min_gap=0.1):
    """One parameter set: (spacing RMSE per segment, failed flag per segment)."""
    rmse, failed = simulate_idm_population({k: [v] for k, v in p.items()}, seg, dt,
                                           collision_gap, min_gap)
    return rmse[0], failed[0]


def objective(x, free, fixed, seg, dt, collision_gap, min_gap, collision_cost):
    """Mean over segments of the spacing RMSE (collision_cost for failed segments).
    x has shape (n_free,) for one parameter set or (n_free, S) for S sets."""
    x = np.asarray(x, dtype=float)
    single = x.ndim == 1
    X = x[:, None] if single else x
    P = {**fixed, **{name: X[i] for i, name in enumerate(free)}}
    rmse, failed = simulate_idm_population(P, seg, dt, collision_gap, min_gap)
    cost = np.mean(np.where(failed, collision_cost, rmse), axis=1)
    return float(cost[0]) if single else cost


def calibrate_idm_sim(seg, cfg_models=None, dt=0.1, collision_gap=0.0):
    """Differential evolution over the IDM bounds. Returns (params, objective, n_evaluations)."""
    cfg_models = cfg_models or config.load_yaml("models.yaml")
    m, sc = cfg_models["idm"], cfg_models["sim_calibration"]
    free, bounds, fixed = split_spec(m["params"])
    opt = sc["optimizer"]
    res = optimize.differential_evolution(
        objective, bounds,
        args=(free, fixed, seg, dt, collision_gap, m["min_gap_m"], sc["collision_cost_m"]),
        popsize=int(opt["popsize"]), maxiter=int(opt["maxiter"]), tol=float(opt["tol"]),
        polish=bool(opt["polish"]), seed=int(sc["seed"]), updating="deferred",
        vectorized=True)
    params = {**fixed, **dict(zip(free, map(float, res.x)))}
    return params, float(res.fun), int(res.nfev)
