"""Statistical comparison of methods (D035).

Unit: the vehicle. Per method, level and context length the run-level spacing RMSE is
averaged over the non-collided runs of a draw, over the draws of a task (as
ema.evaluate.summarize) and over the tasks of a vehicle. Differences of two methods are
paired by vehicle; the uncertainty of their mean comes from a bootstrap over vehicles.
"""
import numpy as np
import pandas as pd

from ema.calibrate import _seed


def vehicle_values(runs):
    """DataFrame level, powertrain, vehicle, length_s, value (mean spacing RMSE)."""
    per_draw = runs.groupby(["level", "task", "draw", "length_s"], dropna=False)["spacing_rmse"].mean()
    per_task = per_draw.groupby(["level", "task", "length_s"], dropna=False).mean().reset_index()
    parts = per_task["task"].str.split(" | ", regex=False)
    per_task["powertrain"], per_task["vehicle"] = parts.str[0], parts.str[1]
    return (per_task.groupby(["level", "powertrain", "vehicle", "length_s"], dropna=False)["spacing_rmse"]
            .mean().rename("value").reset_index())


def at_lengths(values, lengths):
    """Values of a method without context (length NaN, e.g. M0) repeated at every length."""
    if values["length_s"].notna().any():
        return values
    return pd.concat([values.assign(length_s=float(L)) for L in lengths], ignore_index=True)


def paired_differences(a, b, level, lengths):
    """Per vehicle and length: value of a, value of b and d = a - b (only vehicles in both)."""
    a, b = at_lengths(a, lengths), at_lengths(b, lengths)
    keys = ["level", "powertrain", "vehicle", "length_s"]
    j = a[a["level"] == level].merge(b[b["level"] == level], on=keys, suffixes=("_a", "_b"))
    return j.assign(d=j["value_a"] - j["value_b"]).sort_values(keys).reset_index(drop=True)


def bootstrap(d, n_boot, ci, seed):
    """Mean of d, percentile interval of the mean under resampling of its entries, and a
    two-sided bootstrap p-value 2 * (min(#means <= 0, #means >= 0) + 1) / (n_boot + 1)."""
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(int(n_boot), len(d)))].mean(axis=1)
    alpha = (1.0 - float(ci)) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    k = min(int((means <= 0).sum()), int((means >= 0).sum()))
    return float(d.mean()), float(lo), float(hi), min(1.0, 2.0 * (k + 1) / (int(n_boot) + 1))


def holm(p):
    """Holm-adjusted p-values (step-down, monotone, capped at 1), in the input order."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def compare(values, pairs, level, lengths, cfg, kind):
    """One row per pair and length: n vehicles, mean difference (first - second), interval,
    vehicles on which the first is better; p-values only for kind 'confirmatory'."""
    rows = []
    for a, b in pairs:
        if a not in values or b not in values:
            continue
        diff = paired_differences(values[a], values[b], level, lengths)
        for L in lengths:
            d = diff.loc[diff["length_s"] == float(L), "d"].to_numpy()
            if len(d) < int(cfg["min_vehicles_for_intervals"]):
                continue
            seed = int(np.random.SeedSequence(_seed(cfg["seed"], a, b, level, str(float(L))))
                       .generate_state(1)[0])
            mean, lo, hi, p = bootstrap(d, cfg["n_boot"], cfg["ci"], seed)
            rows.append(dict(kind=kind, level=level, method=a, versus=b, length_s=float(L),
                             n_vehicles=len(d), mean_diff=mean, ci_lo=lo, ci_hi=hi,
                             n_better=int((d < 0).sum()),
                             p=p if kind == "confirmatory" else np.nan))
    out = pd.DataFrame(rows)
    if kind == "confirmatory" and len(out):
        out["p_holm"] = holm(out["p"].to_numpy())
    elif len(out):
        out["p_holm"] = np.nan
    return out


def describe(values, pairs, levels, lengths):
    """Vehicle-by-vehicle differences (first - second) for the given levels, no intervals."""
    parts = []
    for a, b in pairs:
        if a not in values or b not in values:
            continue
        for level in levels:
            diff = paired_differences(values[a], values[b], level, lengths)
            if len(diff):
                parts.append(diff.assign(method=a, versus=b))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def collisions(runs_by_method, is_stop):
    """Collided runs per method, level, vehicle and run type (stop / no stop)."""
    parts = []
    for name, runs in runs_by_method.items():
        r = runs.assign(method=name, vehicle=runs["task"].str.split(" | ", regex=False).str[1],
                        run_type=np.where(runs["run_id"].map(is_stop).fillna(False).astype(bool),
                                          "stop", "no_stop"))
        parts.append(r.groupby(["method", "level", "vehicle", "run_type"])
                     .agg(n_runs=("collided", "size"), n_collided=("collided", "sum"))
                     .reset_index())
    out = pd.concat(parts, ignore_index=True)
    return out.assign(share=out["n_collided"] / out["n_runs"])
