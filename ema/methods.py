"""Adaptation methods. Every method follows the same interface:

    method = Method(cfg_models, cfg_eval)
    method.fit(traj_fd, rt, train_run_ids)   # learn from the training runs of a fold only
    method.uses_context                      # False: the context is ignored
    controller = method.controller(context)  # controller for one held-out task and context
    method.state()                           # dict saved with the results (e.g. parameters)
"""
import json

import numpy as np

from ema import calibrate, config, protocol, sim_calibrate, simulate
from ema.calibrate import _seed
from ema.models import idm


class NoAdaptation:
    """M0: one IDM parameter set calibrated on the pooled training runs; no adaptation."""

    name = "no_adaptation"
    uses_context = False

    def __init__(self, cfg_models=None, cfg_eval=None):
        self.cfg = cfg_models or config.load_yaml("models.yaml")
        self.cfg_eval = cfg_eval or config.load_yaml("evaluation.yaml")
        self.params, self.train_rmse, self.train_run_ids = None, None, []

    def fit(self, traj_fd, rt, train_run_ids):
        c = self.cfg["calibration"]
        smp = calibrate.sample_per_task(traj_fd, rt, train_run_ids,
                                        c["samples_per_task"], c["seed"])
        self.params, self.train_rmse = calibrate.calibrate_idm(smp, self.cfg)
        self.train_run_ids = sorted(set(train_run_ids))

    def controller(self, context=None):
        return idm.IDMController(self.params, self.cfg["idm"]["min_gap_m"])

    def state(self):
        return dict(method=self.name, train_rmse=self.train_rmse,
                    n_train_runs=len(self.train_run_ids),
                    **{f"idm_{k}": v for k, v in self.params.items()})


class NoAdaptationSim:
    """M0 with simulation-based calibration (D012): one IDM parameter set fitted in closed
    loop on task-balanced segments of the pooled training runs; no adaptation."""

    name = "no_adaptation_sim"
    uses_context = False

    def __init__(self, cfg_models=None, cfg_eval=None):
        self.cfg = cfg_models or config.load_yaml("models.yaml")
        self.cfg_eval = cfg_eval or config.load_yaml("evaluation.yaml")
        self.params, self.train_cost, self.n_evaluations = None, None, None
        self.n_segments, self.train_run_ids = 0, []

    def fit(self, traj_fd, rt, train_run_ids):
        sim = self.cfg_eval["simulation"]
        if sim.get("spacing_from") != "leader_position":
            raise ValueError("Simulation-based calibration requires spacing_from: leader_position")
        seg = sim_calibrate.make_segments(traj_fd, rt, train_run_ids, self.cfg, sim["dt_s"])
        self.params, self.train_cost, self.n_evaluations = sim_calibrate.calibrate_idm_sim(
            seg, self.cfg, sim["dt_s"], sim["collision_gap_m"])
        self.n_segments = int(len(seg["length"]))
        self.train_run_ids = sorted(set(train_run_ids))

    def controller(self, context=None):
        return idm.IDMController(self.params, self.cfg["idm"]["min_gap_m"])

    def state(self):
        return dict(method=self.name, train_cost=self.train_cost,
                    n_evaluations=self.n_evaluations, n_segments=self.n_segments,
                    n_train_runs=len(self.train_run_ids),
                    **{f"idm_{k}": v for k, v in self.params.items()})


class ContextRecalibration:
    """M1: the IDM is recalibrated in closed loop on the context of the held-out task only
    (same bounds and optimizer as no_adaptation_sim); training runs are not used. A fit
    depends only on (task, draw, context length) and the configuration, so it is cached
    and reused across folds and levels."""

    name = "context_recalibration"
    uses_context = True
    _cache = {}

    def __init__(self, cfg_models=None, cfg_eval=None):
        self.cfg = cfg_models or config.load_yaml("models.yaml")
        self.cfg_eval = cfg_eval or config.load_yaml("evaluation.yaml")
        self.n_new_fits, self.n_cached_fits = 0, 0
        self.train_run_ids = []

    def fit(self, traj_fd, rt, train_run_ids):
        self.train_run_ids = []          # by design, no training data are used

    def _key(self, context):
        cfg_part = json.dumps(dict(idm=self.cfg["idm"], sim=self.cfg["sim_calibration"],
                                   simulation=self.cfg_eval["simulation"],
                                   context=self.cfg_eval.get("context")), sort_keys=True)
        return (context["task"], int(context["draw"]), float(context["length_s"]), cfg_part)

    def _fit(self, context, count=True):
        key = self._key(context)
        if key in self._cache:
            self.n_cached_fits += int(count)
            return self._cache[key]
        sim = self.cfg_eval["simulation"]
        if sim.get("spacing_from") != "leader_position":
            raise ValueError("Simulation-based calibration requires spacing_from: leader_position")
        seg_len = int(round(self.cfg["sim_calibration"]["segment_s"] / sim["dt_s"]))
        pieces = sim_calibrate.cut_into_segments(context["segments"], seg_len)
        seg = sim_calibrate.pack_segments(pieces, sim["dt_s"])
        params, cost, nfev = sim_calibrate.calibrate_idm_sim(seg, self.cfg, sim["dt_s"],
                                                             sim["collision_gap_m"])
        self._cache[key] = dict(params=params, cost=cost, n_evaluations=nfev,
                                n_segments=len(pieces))
        self.n_new_fits += 1
        return self._cache[key]

    def controller(self, context):
        return idm.IDMController(self._fit(context)["params"], self.cfg["idm"]["min_gap_m"])

    def context_info(self, context):
        fit = self._fit(context, count=False)
        return dict(ctx_cost=fit["cost"], ctx_n_segments=fit["n_segments"],
                    **{f"ctx_idm_{k}": v for k, v in fit["params"].items()})

    def state(self):
        return dict(method=self.name, n_train_runs=0, n_new_fits=self.n_new_fits,
                    n_cached_fits=self.n_cached_fits)


def sample_context(pieces, length_s, rng, dt):
    """A context of length_s seconds from a list of run DataFrames: random run order,
    random start in the first run, consecutive runs until the length is reached."""
    order = [pieces[i] for i in rng.permutation(len(pieces))]
    n_need = int(round(length_s / dt)) + 1
    first = order[0]
    start = int(rng.integers(0, max(len(first) - n_need, 0) + 1))
    out, remaining = [], n_need
    for i, p in enumerate(order):
        chunk = p.iloc[(start if i == 0 else 0):][:remaining]
        if len(chunk) >= 2:
            out.append(chunk.reset_index(drop=True))
            remaining -= len(chunk) - 1
        if remaining <= 1:
            break
    return out


class PopulationPrior:
    """M2: the IDM is calibrated on the context of the held-out task with a population
    prior. Prior = mean and spread of the log parameters of the training tasks of the
    fold, each fitted in closed loop on its own training runs (cached; fold-independent).
    The prior weight lambda is chosen per fold by pseudo-tests on training tasks only:
    a context from the training runs of a pseudo-test task, a prior from the other
    training tasks, scored on the validation runs of the pseudo-test task."""

    name = "population_prior"
    uses_context = True
    _task_fits = {}

    def __init__(self, cfg_models=None, cfg_eval=None):
        self.cfg = cfg_models or config.load_yaml("models.yaml")
        self.cfg_eval = cfg_eval or config.load_yaml("evaluation.yaml")
        self.free = calibrate.split_spec(self.cfg["idm"]["params"])[0]
        self.prior, self.lam, self.lambda_scores, self.pseudo_tasks = None, None, {}, []
        self.task_params, self.train_run_ids, self.n_new_task_fits = {}, [], 0
        self._fits = {}

    def _cfg_key(self):
        return json.dumps(dict(idm=self.cfg["idm"], sim=self.cfg["sim_calibration"],
                               simulation=self.cfg_eval["simulation"]), sort_keys=True)

    def _sim(self):
        sim = self.cfg_eval["simulation"]
        if sim.get("spacing_from") != "leader_position":
            raise ValueError("Simulation-based calibration requires spacing_from: leader_position")
        return sim

    def _task_fit(self, traj_fd, rt, task, run_ids):
        key = (task, tuple(sorted(run_ids)), self._cfg_key())
        if key not in self._task_fits:
            sim = self._sim()
            seg = sim_calibrate.make_segments(traj_fd, rt, run_ids, self.cfg, sim["dt_s"])
            params, _, _ = sim_calibrate.calibrate_idm_sim(seg, self.cfg, sim["dt_s"],
                                                           sim["collision_gap_m"])
            self._task_fits[key] = params
            self.n_new_task_fits += 1
        return self._task_fits[key]

    def _map_fit(self, pieces, prior):
        sim = self._sim()
        seg_len = int(round(self.cfg["sim_calibration"]["segment_s"] / sim["dt_s"]))
        seg = sim_calibrate.pack_segments(sim_calibrate.cut_into_segments(pieces, seg_len),
                                          sim["dt_s"])
        params, cost, _ = sim_calibrate.calibrate_idm_sim(seg, self.cfg, sim["dt_s"],
                                                          sim["collision_gap_m"], prior)
        return params, cost

    def fit(self, traj_fd, rt, train_run_ids):
        pp = self.cfg["population_prior"]
        sim = self._sim()
        self.train_run_ids = sorted(set(train_run_ids))
        tr = rt.loc[self.train_run_ids]
        tasks = sorted(tr["task"].unique())
        by_task = {t: sorted(tr.index[tr["task"] == t]) for t in tasks}
        self.task_params = {t: self._task_fit(traj_fd, rt, t, by_task[t]) for t in tasks}
        grid = [float(x) for x in pp["lambdas_m"]]

        def prior_from(task_list, lam):
            return sim_calibrate.make_prior([self.task_params[t] for t in task_list], self.free,
                                            lam, pp["min_log_std"])

        if len(tasks) < int(pp["min_tasks_for_selection"]):
            self.lam = float(np.median(grid))
        else:
            rng = np.random.default_rng(_seed(pp["seed"], *tasks))
            self.pseudo_tasks = sorted(rng.choice(tasks, size=min(int(pp["pseudo_tasks"]),
                                                                  len(tasks)), replace=False))
            runs_by_id = {rid: g.reset_index(drop=True) for rid, g in
                          traj_fd[traj_fd["run_id"].isin(set(rt.index[rt["task"].isin(tasks)]))]
                          .groupby("run_id", sort=False)}
            sums = {lam: [] for lam in grid}
            for j in self.pseudo_tasks:
                val_cfg = protocol.validation_configs(rt, j, self.cfg_eval)
                val_ids = sorted(rt.index[(rt["task"] == j) & rt["config_id"].isin(val_cfg)])
                if not val_ids:
                    continue
                others = [t for t in tasks if t != j]
                for length in pp["pseudo_context_s"]:
                    ctx_rng = np.random.default_rng(_seed(pp["seed"], j, length))
                    ctx = sample_context([runs_by_id[r] for r in by_task[j]], float(length),
                                         ctx_rng, sim["dt_s"])
                    for lam in grid:
                        params, _ = self._map_fit(ctx, prior_from(others, lam))
                        ctrl = idm.IDMController(params, self.cfg["idm"]["min_gap_m"])
                        errs = []
                        for r in val_ids:
                            _, m = simulate.simulate_run(ctrl, runs_by_id[r], self.cfg_eval)
                            errs.append(self.cfg["sim_calibration"]["collision_cost_m"]
                                        if m["collided"] or m["diverged"] else m["spacing_rmse"])
                        sums[lam].append(float(np.mean(errs)))
            self.lambda_scores = {lam: float(np.mean(v)) for lam, v in sums.items() if v}
            self.lam = (min(self.lambda_scores, key=self.lambda_scores.get)
                        if self.lambda_scores else float(np.median(grid)))
        self.prior = prior_from(tasks, self.lam)

    def _fit(self, context):
        key = (context["task"], int(context["draw"]), float(context["length_s"]))
        if key not in self._fits:
            self._fits[key] = self._map_fit(context["segments"], self.prior)
        return self._fits[key]

    def controller(self, context):
        return idm.IDMController(self._fit(context)[0], self.cfg["idm"]["min_gap_m"])

    def context_info(self, context):
        params, cost = self._fit(context)
        return dict(ctx_cost=cost, ctx_lambda=self.lam,
                    **{f"ctx_idm_{k}": v for k, v in params.items()})

    def state(self):
        st = dict(method=self.name, n_train_runs=len(self.train_run_ids),
                  n_train_tasks=len(self.task_params), n_new_task_fits=self.n_new_task_fits,
                  prior_lambda=self.lam, pseudo_tasks="; ".join(self.pseudo_tasks),
                  **{f"lambda_score_{k}": v for k, v in self.lambda_scores.items()})
        for i, k in enumerate(self.free):
            st[f"prior_mu_{k}"] = float(np.exp(self.prior["mu"][i]))
            st[f"prior_logsd_{k}"] = float(self.prior["sd"][i])
        return st


REGISTRY = {m.name: m for m in (NoAdaptation, NoAdaptationSim, ContextRecalibration,
                                PopulationPrior)}
