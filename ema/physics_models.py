"""Alternative physics car-following models for the floor comparison (D028).

Every model has a vectorised one-step rule for the closed-loop calibration (the same
Euler and leader-position simulation as ema.sim_calibrate) and a scalar controller with
the interface of ema.models.idm.IDMController (reset, accel) for ema.simulate.

- idm: the IDM of ema.models.idm (bounds from models.yaml).
- ovrv: linear ACC model a = k1 (s - eta - tau v) + k2 (v_lead - v), clipped to limits.
- idm_lag: the acceleration follows the IDM acceleration with a first-order lag,
  a(k) = a(k-1) + alpha (a_IDM(k) - a(k-1)), alpha = 1 - exp(-dt / tau_a); the initial
  acceleration is the IDM acceleration at the first step. tau_a -> 0 gives the IDM.
"""
import math

import numpy as np
from scipy import optimize

from ema.calibrate import split_spec
from ema.models import idm

MODELS = ("idm", "ovrv", "idm_lag")


def spec(model, cfg_models, cfg_phys):
    """{name: [low, high] or value} of the model's parameters."""
    if model == "idm":
        return dict(cfg_models["idm"]["params"])
    if model == "ovrv":
        return dict(cfg_phys["ovrv"]["params"])
    if model == "idm_lag":
        return {**cfg_models["idm"]["params"], "tau_a": list(cfg_phys["idm_lag"]["tau_a"])}
    raise ValueError(f"unknown model {model}")


def ovrv_accel(s, v, v_lead, p, limits):
    a = p["k1"] * (s - p["eta"] - p["tau"] * v) + p["k2"] * (v_lead - v)
    return np.clip(a, limits[0], limits[1])


def simulate_population(model, P, seg, dt, collision_gap=0.0, min_gap=0.1, limits=(-8.0, 4.0)):
    """As ema.sim_calibrate.simulate_idm_population, for any model of MODELS.
    Returns (spacing RMSE, failed flag), both of shape (S, n_segments)."""
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
    a_prev = None
    if model == "idm_lag":
        alpha = 1.0 - np.exp(-dt / np.asarray(p["tau_a"], dtype=float))
    with np.errstate(all="ignore"):
        for k in range(T - 1):
            active = (~failed) & (k + 1 < length)
            if not active.any():
                break
            if model == "ovrv":
                acc = ovrv_accel(s, v, seg["v_lead"][:, k], p, limits)
            else:
                acc = idm.accel(s, v, seg["v_lead"][:, k], p, min_gap)
                if model == "idm_lag":
                    acc = acc if a_prev is None else a_prev + alpha * (acc - a_prev)
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
            if model == "idm_lag":
                a_prev = np.where(ok, acc, a_prev if a_prev is not None else acc)
    rmse = np.sqrt(err_sum / length)
    return np.where(failed, np.nan, rmse), failed


def calibrate(model, seg, cfg_models, cfg_phys, dt=0.1, collision_gap=0.0):
    """Differential evolution over the model's bounds with the optimizer settings and the
    objective (mean segment RMSE, collision cost for failed segments) of M2's task fit.
    Returns (params, objective)."""
    sc = cfg_models["sim_calibration"]
    free, bounds, fixed = split_spec(spec(model, cfg_models, cfg_phys))
    min_gap = cfg_models["idm"]["min_gap_m"]
    limits = tuple(cfg_phys["ovrv"]["accel_limits"])

    def objective(x):
        X = np.asarray(x, dtype=float)
        X = X[:, None] if X.ndim == 1 else X
        P = {**fixed, **{name: X[i] for i, name in enumerate(free)}}
        rmse, failed = simulate_population(model, P, seg, dt, collision_gap, min_gap, limits)
        cost = np.mean(np.where(failed, sc["collision_cost_m"], rmse), axis=1)
        return float(cost[0]) if np.asarray(x).ndim == 1 else cost

    opt = sc["optimizer"]
    res = optimize.differential_evolution(
        objective, bounds, popsize=int(opt["popsize"]), maxiter=int(opt["maxiter"]),
        tol=float(opt["tol"]), polish=bool(opt["polish"]), seed=int(sc["seed"]),
        updating="deferred", vectorized=True)
    return {**fixed, **dict(zip(free, map(float, res.x)))}, float(res.fun)


class OVRVController:
    """OVRV as a closed-loop controller (scalar arithmetic)."""

    def __init__(self, p, limits=(-8.0, 4.0)):
        self.p = {k: float(p[k]) for k in ("k1", "k2", "eta", "tau")}
        self.lo, self.hi = float(limits[0]), float(limits[1])

    def reset(self, s0, v0, v_lead0):
        pass

    def accel(self, s, v, v_lead):
        p = self.p
        a = p["k1"] * (s - p["eta"] - p["tau"] * v) + p["k2"] * (v_lead - v)
        return min(max(a, self.lo), self.hi)


class IDMLagController:
    """IDM with a first-order actuation lag (scalar arithmetic); see the module notes."""

    def __init__(self, p, min_gap=0.1, dt=0.1):
        self.idm = idm.IDMController(p, min_gap)
        self.alpha = 1.0 - math.exp(-float(dt) / float(p["tau_a"]))
        self.a_prev = None

    def reset(self, s0, v0, v_lead0):
        self.a_prev = None

    def accel(self, s, v, v_lead):
        a_idm = self.idm.accel(s, v, v_lead)
        self.a_prev = a_idm if self.a_prev is None else self.a_prev + self.alpha * (a_idm - self.a_prev)
        return self.a_prev


def controller(model, params, cfg_models, cfg_phys, dt=0.1):
    if model == "idm":
        return idm.IDMController(params, cfg_models["idm"]["min_gap_m"])
    if model == "ovrv":
        return OVRVController(params, cfg_phys["ovrv"]["accel_limits"])
    if model == "idm_lag":
        return IDMLagController(params, cfg_models["idm"]["min_gap_m"], dt)
    raise ValueError(f"unknown model {model}")
