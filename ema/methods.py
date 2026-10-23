"""Adaptation methods. Every method follows the same interface:

    method = Method(cfg_models, cfg_eval)
    method.fit(traj_fd, rt, train_run_ids)   # learn from the training runs of a fold only
    method.uses_context                      # False: the context is ignored
    controller = method.controller(context)  # controller for one held-out task and context
    method.state()                           # dict saved with the results (e.g. parameters)
"""
import json

from ema import calibrate, config, sim_calibrate
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


REGISTRY = {m.name: m for m in (NoAdaptation, NoAdaptationSim, ContextRecalibration)}
