"""Computations behind the group-2 figures (D040 (4), D041 (2) and (4), D050, D053, D054).

Each function recomputes something from the archived results and checks it against the
recorded runs before anything is drawn; a failed check raises ReproductionError. Recorded
CSV files are read with exact float parsing (read_exact, D054).
  gbm_importance      refit features_gbm on the L2 folds; the refit must reproduce the
                      recorded context predictions (ctx_idm_*) exactly; returns the
                      importance of each feature for each IDM parameter, mean over folds
  collision_example   the first L3 collision of m4c_direct_lovo (fixed order), re-simulated
                      with the saved network (same collision, same step) and with the
                      recorded population-prior parameters (same spacing RMSE)
  residual_map        the correction of m2b_residual on a gap x speed-difference grid, mean
                      over the ICE folds of L2; one recorded run per fold is re-simulated
                      with the loaded network (same spacing RMSE)
"""
import numpy as np
import pandas as pd

from ema import calibrate, config, context, data, protocol, simulate
from ema.models import idm

KEYS = ["task", "draw", "length_s", "run_id"]


class ReproductionError(RuntimeError):
    """A recomputation does not reproduce the recorded results."""


def read_exact(path, **kw):
    """Read a recorded CSV with exact float parsing (D054). pandas' default parser can be off
    in the last bit (e.g. 3.1719959010078993 is read as ...8997); targets changed in the last
    bit change the refitted features_gbm by up to 6 % (F046), and every check here compares
    with recorded values exactly or to 1e-9."""
    return pd.read_csv(path, float_precision="round_trip", **kw)


# ---------------------------------------------------------------- protocol

def load_protocol(cfg_e):
    """Run table, folds, run assignment, context table, trajectories (forward-difference
    accelerations, as in scripts/evaluate.py) and the runs by id."""
    rt = protocol.run_table(data.load_runs(), cfg_e)
    folds = protocol.make_folds(rt, cfg_e)
    assignment = protocol.assign_runs(rt, folds, cfg_e)
    ct = context.context_table(rt, cfg_e)
    traj_fd = calibrate.forward_difference(data.load_trajectories(), cfg_e["simulation"]["dt_s"])
    runs_by_id = {rid: g.reset_index(drop=True) for rid, g in traj_fd.groupby("run_id", sort=False)}
    return dict(rt=rt, folds={f["fold_id"]: f for f in folds}, assignment=assignment, ct=ct,
                traj_fd=traj_fd, runs_by_id=runs_by_id)


def make_context(ct, runs_by_id, task, draw, length):
    """The context of one task, draw and length exactly as built by ema.evaluate."""
    from ema import evaluate
    c = ct[(ct["role"] == "context") & (ct["task"] == task) & (ct["draw"] == int(draw))
           & (ct["length_s"] == float(length))]
    if c.empty:
        raise ReproductionError(f"no context rows for {task}, draw {draw}, {length} s")
    return dict(task=task, draw=int(draw), length_s=float(length),
                segments=evaluate.context_data(c, runs_by_id))


def _same(a, b):
    """Exact equality of two floats (NaN equals NaN)."""
    a, b = float(a), float(b)
    return a == b or (np.isnan(a) and np.isnan(b))


def _rel_diff(a, b):
    """Relative difference of two floats (0 if equal, inf if only one is NaN)."""
    a, b = float(a), float(b)
    if a == b or (np.isnan(a) and np.isnan(b)):
        return 0.0
    if np.isnan(a) or np.isnan(b):
        return np.inf
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


# ---------------------------------------------------------------- GBM importance

def floor_targets(idm_runs, param_names):
    """Per task, the cross-configuration IDM fit of diagnostic C (one per task)."""
    d = idm_runs[idm_runs["floor"] == "cross_config"]
    cols = [f"idm_{k}" for k in param_names]
    per = d.groupby("task")[cols]
    if (per.nunique() > 1).any().any():
        raise ValueError("a task has more than one cross-configuration fit")
    return {t: {k: float(r[f"idm_{k}"]) for k in param_names} for t, r in per.first().iterrows()}


def _inject_targets(prot, fold, targets, cfg_m, cfg_e):
    """Put the given task fits into the task-fit cache of PopulationPrior under the keys the
    fold's training would use; returns the keys (to remove them again). D053: the cache is a
    class attribute that outlives earlier calls in the same process, so it is cleared first
    and the fold uses only these fits."""
    from ema import methods
    methods.PopulationPrior._task_fits.clear()
    a = prot["assignment"]
    train = a.loc[(a["fold_id"] == fold["fold_id"]) & (a["role"] == "train"), "run_id"].tolist()
    tr = prot["rt"].loc[sorted(set(train))]
    cfg_key = methods.PopulationPrior(cfg_m, cfg_e)._cfg_key()
    keys = []
    for t in sorted(tr["task"].unique()):
        key = (t, tuple(sorted(tr.index[tr["task"] == t])), cfg_key)
        if key not in methods.PopulationPrior._task_fits:
            methods.PopulationPrior._task_fits[key] = dict(targets[t])
            keys.append(key)
    return keys


def _refit_and_check(prot, fold, recorded, cfg_m, cfg_e, rtol):
    """Refit features_gbm on one fold and compare its context predictions with the recorded
    ctx_idm_* values. Returns (method, n_values_compared, n_not_bit_identical, max relative
    difference, n_beyond_rtol)."""
    from ema import methods
    a = prot["assignment"]
    train = a.loc[(a["fold_id"] == fold["fold_id"]) & (a["role"] == "train"), "run_id"].tolist()
    m = methods.FeaturesGBM(cfg_m, cfg_e)
    m.fit(prot["traj_fd"], prot["rt"], train)
    rec = recorded[recorded["fold_id"] == fold["fold_id"]]
    cols = [c for c in rec.columns if c.startswith("ctx_idm_")]
    n, diff, worst, bad = 0, 0, 0.0, 0
    for (task, draw, length), g in rec.groupby(["task", "draw", "length_s"]):
        info = m.context_info(make_context(prot["ct"], prot["runs_by_id"], task, draw, length))
        r = g.iloc[0]
        for c in cols:
            n += 1
            d = _rel_diff(info[c], r[c])
            diff += d > 0
            worst = max(worst, d)
            bad += d > rtol
    return m, n, diff, worst, bad


def gbm_importance(prot, cfg_m, cfg_e, res_dir, level="L2_new_vehicle", use_floor_targets=True,
                   rtol=1e-9, log=print):
    """Feature importance (impurity based, sums to 1 per parameter) of the refitted
    features_gbm models, mean over the folds of the level; with a reproduction check per
    fold. use_floor_targets (D050): take the task fits from diagnostic C instead of fitting
    again; if the check fails, the fold is refitted with fresh task fits and checked again.
    The check (D051) allows a relative difference of rtol: on one computer, closed-loop fits
    and least squares agree only to about 1e-15 between processes with other thread
    settings (F032), which can move single predictions in the last digits; any real
    difference (other targets, data or settings) is far larger. Not bit-identical values and
    the largest difference are reported. D053: the task-fit cache is cleared before and after,
    and before each fold, so the result does not depend on earlier calls in the process."""
    from ema import context_features, methods
    recorded = read_exact(res_dir / "features_gbm" / "runs.csv")
    free, _, fixed = calibrate.split_spec(cfg_m["idm"]["params"])
    targets = None
    if use_floor_targets:
        fl = floor_targets(read_exact(res_dir / "diagnostics" / "idm_floor" / "idm_floor_runs.csv"), free)
        targets = {t: {**p, **fixed} for t, p in fl.items()}
    imps, checks = [], []
    methods.PopulationPrior._task_fits.clear()             # D053: nothing left from earlier calls
    try:
        for fid in sorted(f for f, fold in prot["folds"].items() if fold["level"] == level):
            fold = prot["folds"][fid]
            source, keys = "fresh fits", []
            if targets is not None:
                keys = _inject_targets(prot, fold, targets, cfg_m, cfg_e)
                source = "diagnostic C fits"
            m, n, diff, worst, bad = _refit_and_check(prot, fold, recorded, cfg_m, cfg_e, rtol)
            log(f"  {fid}: {n} values, {diff} not bit-identical, max relative difference {worst:.2e}, "
                f"{bad} beyond {rtol:g} ({source})")
            if bad and keys:                                   # fall back to fresh task fits
                for k in keys:
                    methods.PopulationPrior._task_fits.pop(k, None)
                source = "fresh fits (diagnostic C fits did not reproduce)"
                m, n, diff, worst, bad = _refit_and_check(prot, fold, recorded, cfg_m, cfg_e, rtol)
                log(f"  {fid}: {n} values, {diff} not bit-identical, max relative difference {worst:.2e}, "
                    f"{bad} beyond {rtol:g} ({source})")
            checks.append(dict(fold_id=fid, targets=source, values=n, not_bit_identical=diff,
                               max_rel_diff=worst, beyond_rtol=bad))
            if bad:
                raise ReproductionError(f"{fid}: refitted features_gbm differs by more than {rtol:g} in {bad} "
                                        f"of {n} recorded context predictions (max {worst:.2e})")
            imps.append(pd.DataFrame({k: m.models[k].feature_importances_ for k in m.free},
                                     index=list(context_features.FEATURE_NAMES)))
    finally:
        methods.PopulationPrior._task_fits.clear()         # nor for later ones
    imp = sum(imps) / len(imps)
    return imp, checks


# ---------------------------------------------------------------- collision example

def first_collision(runs, level="L3_cross_powertrain"):
    """The first collided run of a level in the fixed order task, draw, length, run (D041)."""
    c = runs[(runs["level"] == level) & runs["collided"].astype(bool)]
    if c.empty:
        raise ReproductionError(f"no collision at {level}")
    return c.sort_values(KEYS).iloc[0], len(c)


def collision_example(prot, cfg_m, cfg_e, res_dir, level="L3_cross_powertrain"):
    """Re-simulate the first collision of m4c_direct_lovo at the level with the saved network,
    and the same run with the population prior's recorded context parameters."""
    import torch

    from ema import methods
    runs = read_exact(res_dir / "m4c_direct_lovo" / "runs.csv")
    row, n_coll = first_collision(runs, level)
    task, draw, length, rid, fid = row["task"], int(row["draw"]), float(row["length_s"]), row["run_id"], row["fold_id"]
    ctx = make_context(prot["ct"], prot["runs_by_id"], task, draw, length)
    run = prot["runs_by_id"][rid]

    m = methods.M4DirectLOVO(cfg_m, cfg_e)
    state = torch.load(res_dir / "m4c_direct_lovo" / "nets" / f"{fid}.pt", map_location="cpu")
    m.net.load_state_dict(state)
    m.net = m.net.to(m.cfg_m4.get("eval_device", "cpu")).eval()
    m._cache = {}                                      # predictions of this fold's weights only
    sim_c, met_c = simulate.simulate_run(m.controller(ctx), run, cfg_e)
    if not (met_c["collided"] and met_c["n_valid"] == int(row["n_valid"])):
        raise ReproductionError(f"{fid} {rid}: re-simulation collided={met_c['collided']} at step "
                                f"{met_c['n_valid']}, recorded collided at step {int(row['n_valid'])}")

    pp = read_exact(res_dir / "population_prior" / "runs.csv")
    p_row = pp[(pp["level"] == level) & (pp["task"] == task) & (pp["draw"] == draw)
               & (pp["length_s"] == length) & (pp["run_id"] == rid)]
    if len(p_row) != 1:
        raise ReproductionError("population_prior has no single record of the same run")
    p_row = p_row.iloc[0]
    params = {c[len("ctx_idm_"):]: float(p_row[c]) for c in p_row.index if c.startswith("ctx_idm_")}
    sim_p, met_p = simulate.simulate_run(idm.IDMController(params, cfg_m["idm"]["min_gap_m"]), run, cfg_e)
    if not (_same(met_p["spacing_rmse"], p_row["spacing_rmse"]) and bool(met_p["collided"]) == bool(p_row["collided"])):
        raise ReproductionError(f"population prior re-simulation {met_p['spacing_rmse']} differs from the "
                                f"recorded {p_row['spacing_rmse']}")
    dt = cfg_e["simulation"]["dt_s"]
    check = dict(fold_id=fid, task=task, draw=draw, length_s=length, run_id=rid, collisions_at_level=n_coll,
                 collision_step=int(met_c["n_valid"]), collision_time_s=round((met_c["n_valid"] - 1) * dt, 3),
                 population_prior_spacing_rmse=float(met_p["spacing_rmse"]))
    return dict(t=run["t"].to_numpy(float) - float(run["t"].iloc[0]), s_obs=run["s"].to_numpy(float),
                v_lead=run["v_l"].to_numpy(float), s_net=sim_c["s"], n_valid=int(met_c["n_valid"]),
                s_prior=sim_p["s"], info=check), check


# ---------------------------------------------------------------- residual map

def idm_accel_grid(params, v, gaps, dvs, min_gap):
    """IDM acceleration at follower speed v on the grid gaps x dvs (dv = leader - follower)."""
    c = idm.IDMController(params, min_gap)
    return np.array([[c.accel(s, v, v + dv) for s in gaps] for dv in dvs])


def residual_grid(weights, a_max, scales, params, v, gaps, dvs, min_gap):
    """The correction r (m/s2) of a residual network on the grid (rows dvs, columns gaps),
    with leader and previous acceleration 0 and the IDM acceleration of params."""
    from ema import residual
    G, D = np.meshgrid(gaps, dvs)
    A = idm_accel_grid(params, v, gaps, dvs, min_gap)
    X = np.column_stack([G.ravel(), np.full(G.size, v), D.ravel(), A.ravel(),
                         np.zeros(G.size), np.zeros(G.size)]) / np.asarray(scales, dtype=float)
    return residual.numpy_forward(weights, a_max, X).reshape(G.shape)


def density_level(x, y, xedges, yedges, mass=0.9, smooth=0.0):
    """2-D histogram of the points, optionally Gaussian-smoothed (sigma in bins), and the
    density level whose region holds `mass` of the (smoothed) total. Returns (H with rows y,
    level)."""
    H, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
    if smooth > 0:
        from scipy.ndimage import gaussian_filter
        H = gaussian_filter(H, sigma=float(smooth), mode="constant")
    h = np.sort(H.ravel())[::-1]
    cum = np.cumsum(h) / max(h.sum(), 1e-300)
    level = h[min(np.searchsorted(cum, mass), len(h) - 1)]
    return H.T, float(level)


def residual_map(prot, cfg_m, cfg_e, res_dir, cfg_fig, level="L2_new_vehicle"):
    """Mean correction of m2b_residual over the ICE folds of the level on a grid of gap and
    speed difference (D041 (2), details D050)."""
    import torch

    from ema import residual
    cfg_res = config.load_yaml("residual.yaml")
    rc = cfg_fig["residual_map"]
    dt, min_gap = cfg_e["simulation"]["dt_s"], cfg_m["idm"]["min_gap_m"]
    scales, a_max = residual.scale_vector(cfg_res), float(cfg_res["network"]["a_max"])
    st = read_exact(res_dir / "m2b_residual" / "fold_states.csv")
    st = st[(st["level"] == level) & (st["prior_powertrain"] == "ICE")].sort_values("fold_id")
    runs = read_exact(res_dir / "m2b_residual" / "runs.csv")
    rt, traj = prot["rt"], prot["traj_fd"]
    ice = traj[traj["run_id"].isin(rt.index[rt["powertrain"] == "ICE"])]
    s, v, dv = ice["s"].to_numpy(float), ice["v_f"].to_numpy(float), (ice["v_l"] - ice["v_f"]).to_numpy(float)
    v_med = float(np.median(v))
    q = rc["percentiles"]
    gaps = np.linspace(*np.percentile(s, q), int(rc["n_gap"]))
    dvs = np.linspace(*np.percentile(dv, q), int(rc["n_dv"]))
    free, _, fixed = calibrate.split_spec(cfg_m["idm"]["params"])
    maps, checks = [], []
    for _, f in st.iterrows():
        net = residual.ResidualNet(cfg_res["network"]["hidden"], a_max)
        net.load_state_dict(torch.load(res_dir / "m2b_residual" / "nets" / f"{f['fold_id']}.pt",
                                       map_location="cpu"))
        w = residual.numpy_weights(net)
        rec = runs[(runs["fold_id"] == f["fold_id"]) & ~runs["collided"].astype(bool)].sort_values(KEYS).iloc[0]
        p = {c[len("ctx_idm_"):]: float(rec[c]) for c in rec.index if c.startswith("ctx_idm_")}
        ctrl = residual.ResidualIDMController(p, w, a_max, scales, dt, min_gap)
        _, met = simulate.simulate_run(ctrl, prot["runs_by_id"][rec["run_id"]], cfg_e)
        ok = _same(met["spacing_rmse"], rec["spacing_rmse"])
        checks.append(dict(fold_id=f["fold_id"], run_id=rec["run_id"], recorded=float(rec["spacing_rmse"]),
                           resimulated=float(met["spacing_rmse"]), identical=bool(ok)))
        if not ok:
            raise ReproductionError(f"{f['fold_id']}: re-simulation {met['spacing_rmse']} differs from the "
                                    f"recorded {rec['spacing_rmse']}")
        prior = {**{k: float(f[f"prior_mu_{k}"]) for k in free}, **fixed}
        maps.append(residual_grid(w, a_max, scales, prior, v_med, gaps, dvs, min_gap))
    band = abs(v - v_med) <= float(rc["speed_band_ms"])
    xedges = np.linspace(gaps[0], gaps[-1], int(rc["density_bins"][0]) + 1)    # D051: coarse bins,
    yedges = np.linspace(dvs[0], dvs[-1], int(rc["density_bins"][1]) + 1)     # smoothed
    H, lev = density_level(s[band], dv[band], xedges, yedges, float(rc["mass"]), float(rc["density_smooth"]))
    return dict(gaps=gaps, dvs=dvs, R=np.mean(maps, axis=0), v_med=v_med, H=H, level=lev,
                hx=(xedges[:-1] + xedges[1:]) / 2, hy=(yedges[:-1] + yedges[1:]) / 2,
                n_folds=len(maps), n_states=int(band.sum())), checks
