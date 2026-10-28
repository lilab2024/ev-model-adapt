"""Hand-crafted context features for features_gbm (D030).

context_features turns a context (pieces with columns s, v_f, v_l, sampled at dt) into a
fixed vector of FEATURE_NAMES; every entry is finite (fallbacks when a statistic is
undefined, e.g. no sample above the speed threshold).
"""
import numpy as np

FEATURE_NAMES = ("gap_intercept", "gap_slope", "tg_mean", "tg_median", "tg_q10", "tg_q90",
                 "acc_std", "acc_q05", "acc_q95", "acc_min", "k_gap", "k_dv", "v_mean",
                 "v_range", "dv_mean", "stop_share", "length_s", "is_ev")


def _arrays(piece):
    """(s, v_f, v_l) as float arrays from a DataFrame or a dict of arrays."""
    return tuple(np.asarray(piece[c], dtype=float) for c in ("s", "v_f", "v_l"))


def context_features(pieces, dt, is_ev, tg_min_speed=5.0, stop_speed=0.5):
    """Feature vector (len(FEATURE_NAMES),) of a context."""
    s_l, v_l_, dv_l, a_l, s_a, v_a, dv_a = [], [], [], [], [], [], []
    n_samples = 0
    for p in pieces:
        s, v, vl = _arrays(p)
        if len(s) < 2:
            continue
        n_samples += len(s) - 1
        s_l.append(s)
        v_l_.append(v)
        dv_l.append(vl - v)
        a = np.diff(v) / dt                                    # forward difference
        a_l.append(a)
        s_a.append(s[:-1])
        v_a.append(v[:-1])
        dv_a.append((vl - v)[:-1])
    s, v, dv = np.concatenate(s_l), np.concatenate(v_l_), np.concatenate(dv_l)
    a, sa, va, dva = (np.concatenate(x) for x in (a_l, s_a, v_a, dv_a))

    # straight-line fit gap = c0 + c1 v (falls back to a proportional fit if v hardly varies)
    if np.ptp(v) > 0.5:
        c1, c0 = np.polyfit(v, s, 1)
    else:
        c1, c0 = float(np.mean(s) / max(np.mean(v), 1.0)), 0.0
    fast = v > tg_min_speed
    tg = s[fast] / v[fast] if fast.any() else np.array([np.mean(s) / max(np.mean(v), 1.0)])
    # linear reaction a = k_gap (gap - c0 - c1 v) + k_dv (speed difference), no intercept
    X = np.column_stack([sa - c0 - c1 * va, dva])
    k = np.linalg.lstsq(X, a, rcond=None)[0] if len(a) >= 2 else np.zeros(2)
    out = np.array([c0, c1, tg.mean(), np.median(tg), np.quantile(tg, 0.1), np.quantile(tg, 0.9),
                    a.std(), np.quantile(a, 0.05), np.quantile(a, 0.95), a.min(), k[0], k[1],
                    v.mean(), np.ptp(v), dv.mean(), float(np.mean(v < stop_speed)),
                    n_samples * dt, float(bool(is_ev))])
    return np.where(np.isfinite(out), out, 0.0)
