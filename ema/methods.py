"""Adaptation methods. Every method follows the same interface:

    method = Method(cfg_models, cfg_eval)
    method.fit(traj_fd, rt, train_run_ids)   # learn from the training runs of a fold only
    method.uses_context                      # False: the context is ignored
    controller = method.controller(context)  # controller for one held-out task and context
    method.state()                           # dict saved with the results (e.g. parameters)
"""
import hashlib
import json
from pathlib import Path

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

    def _effective_prior(self, pieces, prior):
        """Prior weight scaled by reference_context_s / context length (more data, weaker prior)."""
        dt = self._sim()["dt_s"]
        length = sum(len(p) - 1 for p in pieces) * dt
        scale = float(self.cfg["population_prior"]["reference_context_s"]) / max(length, dt)
        return {**prior, "lam": prior["lam"] * scale}

    def _map_fit(self, pieces, prior):
        """Context fit with the length-scaled prior. Returns (params, cost, effective lambda)."""
        sim = self._sim()
        seg_len = int(round(self.cfg["sim_calibration"]["segment_s"] / sim["dt_s"]))
        seg = sim_calibrate.pack_segments(sim_calibrate.cut_into_segments(pieces, seg_len),
                                          sim["dt_s"])
        eff = self._effective_prior(pieces, prior)
        params, cost, _ = sim_calibrate.calibrate_idm_sim(seg, self.cfg, sim["dt_s"],
                                                          sim["collision_gap_m"], eff)
        return params, cost, eff["lam"]

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
            self.lam = float(pp["default_lambda_m"])
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
                        params, _, _ = self._map_fit(ctx, prior_from(others, lam))
                        ctrl = idm.IDMController(params, self.cfg["idm"]["min_gap_m"])
                        errs = []
                        for r in val_ids:
                            _, m = simulate.simulate_run(ctrl, runs_by_id[r], self.cfg_eval)
                            errs.append(self.cfg["sim_calibration"]["collision_cost_m"]
                                        if m["collided"] or m["diverged"] else m["spacing_rmse"])
                        sums[lam].append(float(np.mean(errs)))
            self.lambda_scores = {lam: float(np.mean(v)) for lam, v in sums.items() if v}
            self.lam = (min(self.lambda_scores, key=self.lambda_scores.get)
                        if self.lambda_scores else float(pp["default_lambda_m"]))
        self.prior = prior_from(tasks, self.lam)

    def _fit(self, context):
        key = (context["task"], int(context["draw"]), float(context["length_s"]))
        if key not in self._fits:
            self._fits[key] = self._map_fit(context["segments"], self.prior)
        return self._fits[key]

    def controller(self, context):
        return idm.IDMController(self._fit(context)[0], self.cfg["idm"]["min_gap_m"])

    def context_info(self, context):
        params, cost, lam_eff = self._fit(context)
        return dict(ctx_cost=cost, ctx_lambda=self.lam, ctx_lambda_eff=lam_eff,
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


def load_pretrained_net(cfg_m4, cfg_pre, cfg_models, device="cpu"):
    """The pretrained M4 network (D019). The weights file must match the md5 in
    config/m4.yaml, and the parameter bounds stored with the weights must equal the
    bounds in config/models.yaml. Returns (network in eval mode, md5)."""
    import torch

    from ema import nets
    path = Path(cfg_m4["pretrained"]["weights"])
    if not path.is_absolute():
        path = config.REPO_ROOT / path
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    if md5 != cfg_m4["pretrained"]["md5"]:
        raise ValueError(f"{path}: md5 {md5} does not match config/m4.yaml "
                         f"({cfg_m4['pretrained']['md5']})")
    net = nets.AdaptNet(cfg_pre["network"], cfg_models["idm"]["params"])
    state = torch.load(path, map_location="cpu")
    for k in ("log_lo", "log_hi"):
        if not torch.allclose(state[k], getattr(net, k)):
            raise ValueError(f"{path}: parameter bounds of the saved network differ from "
                             "config/models.yaml")
    net.load_state_dict(state)
    return net.to(device).eval(), md5


class M4Pretrained:
    """M4-A without fine-tuning (D019, step 7c-1): the network pretrained on synthetic
    vehicles turns the context into IDM parameters; no real training data are used.
    The context is cut into pieces of at most context_piece_s (config/m4.yaml); the
    encoder averages the pieces. A prediction depends only on the context and the
    weights, so it is cached and reused across folds and levels."""

    name = "m4a_pretrained"
    uses_context = True
    _cache = {}

    def __init__(self, cfg_models=None, cfg_eval=None, cfg_m4=None, cfg_pre=None):
        self.cfg = cfg_models or config.load_yaml("models.yaml")
        self.cfg_eval = cfg_eval or config.load_yaml("evaluation.yaml")
        self.cfg_m4 = cfg_m4 or config.load_yaml("m4.yaml")
        self.cfg_pre = cfg_pre or config.load_yaml("pretraining.yaml")
        self.net, self.weights_md5 = load_pretrained_net(
            self.cfg_m4, self.cfg_pre, self.cfg, self.cfg_m4.get("eval_device", "cpu"))
        self.n_new, self.n_cached, self.train_run_ids = 0, 0, []

    def fit(self, traj_fd, rt, train_run_ids):
        self.train_run_ids = []          # by design, no real training data are used

    def context_pieces(self, segments):
        """Context DataFrames cut into pieces of at most context_piece_s."""
        dt = self.cfg_eval["simulation"]["dt_s"]
        return sim_calibrate.cut_into_segments(
            segments, int(round(float(self.cfg_m4["context_piece_s"]) / dt)))

    def _key(self, context):
        cfg_part = json.dumps(dict(idm=self.cfg["idm"], network=self.cfg_pre["network"],
                                   piece_s=self.cfg_m4["context_piece_s"],
                                   simulation=self.cfg_eval["simulation"],
                                   context=self.cfg_eval.get("context")), sort_keys=True)
        return (self.weights_md5, context["task"], int(context["draw"]),
                float(context["length_s"]), cfg_part)

    def _predict(self, context, count=True):
        from ema import train_m4
        key = self._key(context)
        if key in self._cache:
            self.n_cached += int(count)
            return self._cache[key]
        pieces = self.context_pieces(context["segments"])
        params = train_m4.predict_params(self.net, pieces, self.cfg_eval["simulation"]["dt_s"],
                                         self.cfg_pre["network"]["scales"])
        self._cache[key] = dict(params=params, n_pieces=len(pieces))
        self.n_new += 1
        return self._cache[key]

    def controller(self, context):
        return idm.IDMController(self._predict(context)["params"], self.cfg["idm"]["min_gap_m"])

    def context_info(self, context):
        pred = self._predict(context, count=False)
        return dict(ctx_n_pieces=pred["n_pieces"],
                    **{f"ctx_idm_{k}": v for k, v in pred["params"].items()})

    def state(self):
        return dict(method=self.name, n_train_runs=0, weights_md5=self.weights_md5,
                    n_new_predictions=self.n_new, n_cached_predictions=self.n_cached)


class M4Finetuned(M4Pretrained):
    """M4-A fine-tuned per fold (D019, D020, step 7c-2): the pretrained network is trained
    further on the fold's training tasks. Training episodes: task-balanced, context and
    60 s targets from different training runs of a task. Fixed validation episodes:
    context from training runs, targets from validation runs of the same task; early
    stopping on them, with the starting (pretrained) weights kept unless fine-tuning
    improves the validation loss. Contexts are cut into pieces of at most context_piece_s,
    as at test time. Seeds derive from the fold's training-task list."""

    name = "m4a_finetuned"

    def __init__(self, cfg_models=None, cfg_eval=None, cfg_m4=None, cfg_pre=None, log=print):
        super().__init__(cfg_models, cfg_eval, cfg_m4, cfg_pre)
        self._cache = {}                 # predictions depend on this fold's weights
        self.log, self.history, self.info = log, [], {}

    def fit(self, traj_fd, rt, train_run_ids):
        import time

        import torch

        from ema import train_m4
        ft = self.cfg_m4["finetuning"]
        dt = self.cfg_eval["simulation"]["dt_s"]
        piece_len = int(round(float(self.cfg_m4["context_piece_s"]) / dt))
        self.train_run_ids = sorted(set(train_run_ids))
        tr = rt.loc[self.train_run_ids]
        tasks = sorted(tr["task"].unique())
        train_runs = {t: sorted(tr.index[tr["task"] == t]) for t in tasks}
        val_runs = {}
        for t in tasks:
            cfgs = protocol.validation_configs(rt, t, self.cfg_eval)
            val_runs[t] = sorted(rt.index[(rt["task"] == t) & rt["config_id"].isin(cfgs)])
        val_ids = sorted(r for v in val_runs.values() for r in v)
        if set(val_ids) & set(self.train_run_ids):
            raise RuntimeError("validation runs overlap with training runs")
        fold_seed = int(np.random.SeedSequence(_seed(ft["seed"], *tasks)).generate_state(1)[0])
        cfg_tr = dict(context_s=list(ft["context_s"]), target_s=float(ft["target_s"]),
                      targets_per_episode=int(ft["targets_per_episode"]),
                      batch_episodes=int(ft["batch_episodes"]), steps=int(ft["max_steps"]),
                      lr=float(ft["lr"]), grad_clip=float(ft["grad_clip"]),
                      val_every=int(ft["val_every"]), patience=int(ft["patience"]),
                      seed=fold_seed)
        arrays = train_m4.run_arrays(traj_fd, self.train_run_ids + val_ids)
        sampler = train_m4.EpisodeSampler(arrays, train_runs, cfg_tr, dt, fold_seed,
                                          piece_len=piece_len)
        items = train_m4.validation_episodes(arrays, train_runs, val_runs, ft["context_s"],
                                             ft["val_episodes_per_task_and_length"], cfg_tr,
                                             dt, fold_seed + 1, piece_len=piece_len)
        if not items:
            raise RuntimeError("no validation episodes (no training task has validation runs)")
        device = train_m4.resolve_device(ft.get("device", "cpu"))
        self.net = self.net.to(device)
        t0 = time.time()
        self.history = train_m4.train(self.net, sampler, [e for _, _, e in items], cfg_tr, dt,
                                      self.cfg_pre["network"]["scales"], log=self.log,
                                      min_gap=self.cfg["idm"]["min_gap_m"], eval_at_start=True)
        self.net = self.net.to(self.cfg_m4.get("eval_device", "cpu")).eval()
        best = min(self.history, key=lambda h: h["val_loss"])
        self.info = dict(n_val_runs=len(val_ids), n_val_episodes=len(items),
                         n_train_tasks_sampled=len(sampler.tasks), fold_seed=fold_seed,
                         val_loss_start=self.history[0]["val_loss"], val_loss_best=best["val_loss"],
                         best_step=int(best["step"]), steps_run=int(self.history[-1]["step"]),
                         n_skipped_steps=int(self.history[-1]["skipped"]),
                         finetune_s=round(time.time() - t0, 1), device=str(device),
                         gpu_name=(torch.cuda.get_device_name(device) if device.type == "cuda"
                                   else "none"),
                         torch_version=torch.__version__,
                         torch_threads=torch.get_num_threads())
        self._cache = {}

    def state(self):
        return dict(super().state(), n_train_runs=len(self.train_run_ids), **self.info)

    def save_artifacts(self, directory, fold_id):
        """Fine-tuned weights (nets/<fold>.pt) and training history (histories/<fold>.csv)."""
        import pandas as pd
        import torch

        from ema import train_m4
        directory = Path(directory)
        (directory / "nets").mkdir(parents=True, exist_ok=True)
        (directory / "histories").mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().cpu() for k, v in self.net.state_dict().items()}
        train_m4._atomic_write(directory / "nets" / f"{fold_id}.pt",
                               lambda p: torch.save(state, p))
        train_m4._atomic_write(directory / "histories" / f"{fold_id}.csv",
                               lambda p: pd.DataFrame(self.history).to_csv(p, index=False))


class M4FinetunedLOVO(M4Finetuned):
    """POST HOC variant of m4a_finetuned (D024; designed after seeing the test results of
    F030). Only the choice of the number of fine-tuning steps differs: stage 1 fine-tunes
    on the training vehicles except one held-out vehicle (preferably of the test vehicle's
    powertrain) with early stopping on validation episodes of that vehicle only, giving the
    best step S; stage 2 fine-tunes again from the pretrained network on all training
    vehicles for exactly S steps (same seeds and sampler as m4a_finetuned, so its weights
    are those of m4a_finetuned after S steps, up to GPU rounding). Only for the new-vehicle
    levels; with fewer than 2 training vehicles the original validation is used."""

    name = "m4a_finetuned_lovo"
    LEVELS = ("L2_new_vehicle", "L2_no_ice", "L3_cross_powertrain")

    def set_fold(self, fold):
        self.fold = fold

    def fit(self, traj_fd, rt, train_run_ids):
        import copy
        import time

        import torch

        from ema import train_m4
        fold = getattr(self, "fold", None)
        if fold is None or fold["level"] not in self.LEVELS:
            raise ValueError("m4a_finetuned_lovo runs only at the levels " + ", ".join(self.LEVELS))
        ft = self.cfg_m4["finetuning"]
        dt = self.cfg_eval["simulation"]["dt_s"]
        piece_len = int(round(float(self.cfg_m4["context_piece_s"]) / dt))
        ids = sorted(set(train_run_ids))
        tr = rt.loc[ids]
        tasks = sorted(tr["task"].unique())
        vehicle = {t: str(rt.loc[rt["task"] == t, "vehicle"].iloc[0]) for t in tasks}
        powertrain = {t: str(rt.loc[rt["task"] == t, "powertrain"].iloc[0]) for t in tasks}
        vehicles = sorted(set(vehicle.values()))
        if len(vehicles) < 2:                            # no hold-out possible: original method
            super().fit(traj_fd, rt, train_run_ids)
            self.info.update(lovo_used=False, heldout_vehicle="none", heldout_same_powertrain=False)
            self.history2 = []
            return
        test_pt = str(rt.loc[rt["task"] == fold["test"][0], "powertrain"].iloc[0])
        fold_seed = int(np.random.SeedSequence(_seed(ft["seed"], *tasks)).generate_state(1)[0])
        train_runs = {t: sorted(tr.index[tr["task"] == t]) for t in tasks}
        val_runs = {}
        for t in tasks:
            cfgs = protocol.validation_configs(rt, t, self.cfg_eval)
            val_runs[t] = sorted(rt.index[(rt["task"] == t) & rt["config_id"].isin(cfgs)])
        val_ids = sorted(r for v in val_runs.values() for r in v)
        if set(val_ids) & set(ids):
            raise RuntimeError("validation runs overlap with training runs")
        cfg_tr = dict(context_s=list(ft["context_s"]), target_s=float(ft["target_s"]),
                      targets_per_episode=int(ft["targets_per_episode"]),
                      batch_episodes=int(ft["batch_episodes"]), steps=int(ft["max_steps"]),
                      lr=float(ft["lr"]), grad_clip=float(ft["grad_clip"]),
                      val_every=int(ft["val_every"]), patience=int(ft["patience"]),
                      seed=fold_seed)
        arrays = train_m4.run_arrays(traj_fd, ids + val_ids)

        # held-out vehicle: same powertrain as the test vehicle if possible, random with the
        # fold seed; the first candidate (in that random order) with validation episodes
        same = sorted(v for v in vehicles if any(powertrain[t] == test_pt
                                                 for t in tasks if vehicle[t] == v))
        candidates = same if same else vehicles
        order = [candidates[i] for i in np.random.default_rng(fold_seed + 2).permutation(len(candidates))]
        held, items = None, []
        for v in order:
            held_tasks = [t for t in tasks if vehicle[t] == v]
            items = train_m4.validation_episodes(
                arrays, {t: train_runs[t] for t in held_tasks}, {t: val_runs[t] for t in held_tasks},
                ft["context_s"], ft["val_episodes_per_task_and_length"], cfg_tr, dt, fold_seed + 1,
                piece_len=piece_len)
            if items:
                held = v
                break
        if held is None:                                 # no usable hold-out: original method
            super().fit(traj_fd, rt, train_run_ids)
            self.info.update(lovo_used=False, heldout_vehicle="none", heldout_same_powertrain=False)
            self.history2 = []
            return
        stage1_runs = {t: r for t, r in train_runs.items() if vehicle[t] != held}
        device = train_m4.resolve_device(ft.get("device", "cpu"))
        scales, min_gap = self.cfg_pre["network"]["scales"], self.cfg["idm"]["min_gap_m"]
        t0 = time.time()

        # stage 1: choose S on the held-out vehicle
        net1 = copy.deepcopy(self.net).to(device)
        sampler1 = train_m4.EpisodeSampler(arrays, stage1_runs, cfg_tr, dt, fold_seed,
                                           piece_len=piece_len)
        self.log(f"stage 1: held-out vehicle {held} ({'same' if same else 'other'} powertrain)")
        self.history = train_m4.train(net1, sampler1, [e for _, _, e in items], cfg_tr, dt, scales,
                                      log=self.log, min_gap=min_gap, eval_at_start=True)
        best = min(self.history, key=lambda h: h["val_loss"])
        s_best = int(best["step"])
        del net1

        # stage 2: all training vehicles, exactly S steps from the pretrained network
        self.net = self.net.to(device)
        sampler2 = train_m4.EpisodeSampler(arrays, train_runs, cfg_tr, dt, fold_seed,
                                           piece_len=piece_len)
        self.history2 = train_m4.train_fixed(self.net, sampler2, dict(cfg_tr, steps=s_best), dt,
                                             scales, log=self.log, min_gap=min_gap)
        self.net = self.net.to(self.cfg_m4.get("eval_device", "cpu")).eval()
        self.train_run_ids = ids
        self.info = dict(lovo_used=True, heldout_vehicle=held, heldout_same_powertrain=bool(same),
                         heldout_tasks=[t for t in tasks if vehicle[t] == held],
                         n_val_runs=sum(len(val_runs[t]) for t in tasks if vehicle[t] == held),
                         n_val_episodes=len(items), n_train_tasks_stage1=len(sampler1.tasks),
                         n_train_tasks_sampled=len(sampler2.tasks), fold_seed=fold_seed,
                         val_loss_start=self.history[0]["val_loss"], val_loss_best=best["val_loss"],
                         best_step=s_best, steps_run=int(self.history[-1]["step"]),
                         stage2_steps=s_best,
                         n_skipped_steps=int(self.history[-1]["skipped"])
                         + int(self.history2[-1]["skipped"] if self.history2 else 0),
                         finetune_s=round(time.time() - t0, 1), device=str(device),
                         gpu_name=(torch.cuda.get_device_name(device) if device.type == "cuda"
                                   else "none"),
                         torch_version=torch.__version__, torch_threads=torch.get_num_threads())
        self._cache = {}

    def save_artifacts(self, directory, fold_id):
        """As m4a_finetuned (stage-2 weights; histories/<fold>.csv is the stage-1 curve), plus
        histories/<fold>_stage2.csv."""
        import pandas as pd

        from ema import train_m4
        super().save_artifacts(directory, fold_id)
        train_m4._atomic_write(Path(directory) / "histories" / f"{fold_id}_stage2.csv",
                               lambda p: pd.DataFrame(self.history2).to_csv(p, index=False))


class PopulationPriorSamePowertrain(PopulationPrior):
    """M2b (planned in the research plan before the M4 results; details in D025): M2 with
    the prior built only from training tasks of the test vehicle's powertrain; everything
    else (task fits, pseudo-tests for lambda, context fit) as in M2. Defined for L1 and L2.
    At L2_no_ice it would equal M2 (all training vehicles are EVs); at L3 there is no
    training vehicle of the test powertrain."""

    name = "population_prior_same_powertrain"
    LEVELS = ("L1_new_setting", "L2_new_vehicle")

    def set_fold(self, fold):
        self.fold = fold

    def fit(self, traj_fd, rt, train_run_ids):
        fold = getattr(self, "fold", None)
        if fold is None or fold["level"] not in self.LEVELS:
            raise ValueError("population_prior_same_powertrain runs only at the levels "
                             + ", ".join(self.LEVELS))
        self.powertrain = str(rt.loc[rt["task"] == fold["test"][0], "powertrain"].iloc[0])
        keep = [r for r in train_run_ids if str(rt.loc[r, "powertrain"]) == self.powertrain]
        if not keep:
            raise ValueError(f"no training runs of powertrain {self.powertrain}")
        super().fit(traj_fd, rt, keep)

    def state(self):
        return dict(super().state(), method=self.name, prior_powertrain=self.powertrain)


class M2bResidual(PopulationPriorSamePowertrain):
    """m2b_residual (D027, POST HOC): IDM parameters exactly as M2b (same task fits,
    lambda, prior and context fits), plus a bounded learned acceleration correction
    (ema.residual) trained on the training vehicles of the test powertrain, each window
    with its task's own fit. The number of steps S comes from stage 1 (one held-out
    vehicle of the test powertrain, early stopping on its validation runs); stage 2 trains
    a new network from the same initialisation on all of them for exactly S steps. With
    fewer than 2 such vehicles, stage 1 validates on the validation runs of all of them.
    With S = 0 the method equals M2b."""

    name = "m2b_residual"

    def __init__(self, cfg_models=None, cfg_eval=None, cfg_res=None, log=print):
        super().__init__(cfg_models, cfg_eval)
        self.cfg_res = cfg_res or config.load_yaml("residual.yaml")
        self.log, self.weights, self.history, self.history2, self.res_info = log, None, [], [], {}

    def fit(self, traj_fd, rt, train_run_ids):
        import time

        import torch

        from ema import residual, train_m4
        super().fit(traj_fd, rt, train_run_ids)          # M2b: powertrain, task fits, prior
        cr, tr_cfg = self.cfg_res, self.cfg_res["training"]
        dt = self.cfg_eval["simulation"]["dt_s"]
        min_gap = self.cfg["idm"]["min_gap_m"]
        scales = residual.scale_vector(cr)
        tasks = sorted(self.task_params)
        ids = set(self.train_run_ids)
        train_runs = {t: sorted(r for r in rt.index[rt["task"] == t] if r in ids) for t in tasks}
        val_runs = {}
        for t in tasks:
            cfgs = protocol.validation_configs(rt, t, self.cfg_eval)
            val_runs[t] = sorted(rt.index[(rt["task"] == t) & rt["config_id"].isin(cfgs)])
        if set(r for v in val_runs.values() for r in v) & ids:
            raise RuntimeError("validation runs overlap with training runs")
        vehicle = {t: str(rt.loc[rt["task"] == t, "vehicle"].iloc[0]) for t in tasks}
        vehicles = sorted(set(vehicle.values()))
        fold_seed = int(np.random.SeedSequence(_seed(tr_cfg["seed"], *tasks)).generate_state(1)[0])
        arrays = train_m4.run_arrays(traj_fd, sorted(ids | {r for v in val_runs.values() for r in v}))
        n_win = int(round(float(tr_cfg["window_s"]) / dt)) + 1

        def val_items(val_tasks, seed):
            rng, items = np.random.default_rng(seed), []
            for t in val_tasks:
                for _ in range(int(tr_cfg["val_windows_per_task"]) if val_runs[t] else 0):
                    a = arrays[val_runs[t][int(rng.integers(len(val_runs[t])))]]
                    start = int(rng.integers(0, max(len(a["s"]) - n_win, 0) + 1))
                    items.append((t, {c: a[c][start:start + n_win] for c in ("s", "v_f", "v_l")}))
            return items

        held = None
        if len(vehicles) >= 2:
            order = [vehicles[i] for i in np.random.default_rng(fold_seed + 2).permutation(len(vehicles))]
            for v in order:
                if any(val_runs[t] for t in tasks if vehicle[t] == v):
                    held = v
                    break
        stage1_tasks = [t for t in tasks if vehicle[t] != held]
        val_tasks = [t for t in tasks if vehicle[t] == held] if held else tasks
        device = train_m4.resolve_device(tr_cfg.get("device", "cpu"))
        net_cfg = cr["network"]
        t0 = time.time()

        torch.manual_seed(fold_seed)
        net1 = residual.ResidualNet(net_cfg["hidden"], net_cfg["a_max"]).to(device)
        sampler1 = residual.WindowSampler(arrays, {t: train_runs[t] for t in stage1_tasks},
                                          tr_cfg["window_s"], dt, fold_seed)
        items = val_items(val_tasks, fold_seed + 1)
        if not items:
            raise RuntimeError("no validation windows for the residual")
        self.log(f"residual stage 1: held-out vehicle {held}, {len(stage1_tasks)} training tasks")
        self.history = residual.train(net1, sampler1, items, self.task_params, tr_cfg, dt, scales,
                                      int(tr_cfg["max_steps"]), True, self.log, min_gap)
        best = min(self.history, key=lambda h: h["val_loss"])
        s_best = int(best["step"])
        del net1

        torch.manual_seed(fold_seed)
        net = residual.ResidualNet(net_cfg["hidden"], net_cfg["a_max"]).to(device)
        sampler2 = residual.WindowSampler(arrays, train_runs, tr_cfg["window_s"], dt, fold_seed)
        self.history2 = residual.train(net, sampler2, [], self.task_params, tr_cfg, dt, scales,
                                       s_best, False, self.log, min_gap)
        self.net_state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        self.weights = residual.numpy_weights(net.cpu())
        self.res_info = dict(res_heldout_vehicle=held or "none", res_stage1_tasks=len(stage1_tasks),
                             res_val_windows=len(items), res_val_loss_start=self.history[0]["val_loss"],
                             res_val_loss_best=best["val_loss"], res_best_step=s_best,
                             res_steps_run=int(self.history[-1]["step"]), res_stage2_steps=s_best,
                             res_fold_seed=fold_seed, res_train_s=round(time.time() - t0, 1),
                             res_device=str(device),
                             res_gpu=(torch.cuda.get_device_name(device) if device.type == "cuda"
                                      else "none"))

    def controller(self, context):
        from ema import residual
        params = self._fit(context)[0]
        return residual.ResidualIDMController(params, self.weights, self.cfg_res["network"]["a_max"],
                                              residual.scale_vector(self.cfg_res),
                                              self.cfg_eval["simulation"]["dt_s"],
                                              self.cfg["idm"]["min_gap_m"])

    def state(self):
        return dict(super().state(), method=self.name, **self.res_info)

    def save_artifacts(self, directory, fold_id):
        """Correction weights (nets/<fold>.pt) and the stage-1 and stage-2 histories."""
        import pandas as pd
        import torch

        from ema import train_m4
        directory = Path(directory)
        (directory / "nets").mkdir(parents=True, exist_ok=True)
        (directory / "histories").mkdir(parents=True, exist_ok=True)
        train_m4._atomic_write(directory / "nets" / f"{fold_id}.pt",
                               lambda p: torch.save(self.net_state, p))
        train_m4._atomic_write(directory / "histories" / f"{fold_id}.csv",
                               lambda p: pd.DataFrame(self.history).to_csv(p, index=False))
        train_m4._atomic_write(directory / "histories" / f"{fold_id}_stage2.csv",
                               lambda p: pd.DataFrame(self.history2).to_csv(p, index=False))


class FeaturesGBM:
    """features_gbm (D030, POST HOC): hand-crafted context features (ema.context_features)
    and one gradient-boosting regressor per free IDM parameter (log scale), trained per
    fold on random contexts of all training tasks with each task's own fit (M2's task fit)
    as target; the prediction (clipped to the bounds) drives the IDM in closed loop."""

    name = "features_gbm"
    uses_context = True

    def __init__(self, cfg_models=None, cfg_eval=None, cfg_fg=None):
        self.cfg = cfg_models or config.load_yaml("models.yaml")
        self.cfg_eval = cfg_eval or config.load_yaml("evaluation.yaml")
        self.cfg_fg = cfg_fg or config.load_yaml("features_gbm.yaml")
        self.free, self.bounds, self.fixed = calibrate.split_spec(self.cfg["idm"]["params"])
        self.models, self.train_run_ids, self.info, self._cache = {}, [], {}, {}

    def _features(self, pieces, powertrain):
        from ema import context_features
        f = self.cfg_fg["features"]
        return context_features.context_features(pieces, self.cfg_eval["simulation"]["dt_s"],
                                                 powertrain == "EV", f["tg_min_speed"],
                                                 f["stop_speed"])

    def fit(self, traj_fd, rt, train_run_ids):
        from sklearn.ensemble import GradientBoostingRegressor

        from ema import train_m4
        fg = self.cfg_fg
        dt = self.cfg_eval["simulation"]["dt_s"]
        self.train_run_ids = sorted(set(train_run_ids))
        tr = rt.loc[self.train_run_ids]
        tasks = sorted(tr["task"].unique())
        runs = {t: sorted(tr.index[tr["task"] == t]) for t in tasks}
        helper = PopulationPrior(self.cfg, self.cfg_eval)
        task_params = {t: helper._task_fit(traj_fd, rt, t, runs[t]) for t in tasks}
        seed = int(np.random.SeedSequence(_seed(fg["seed"], *tasks)).generate_state(1)[0])
        rng = np.random.default_rng(seed)
        arrays = train_m4.run_arrays(traj_fd, self.train_run_ids)
        X, Y = [], []
        for t in tasks:
            pt = str(rt.loc[runs[t][0], "powertrain"])
            for length in fg["context_s"]:
                for _ in range(int(fg["contexts_per_task_length"])):
                    order = [runs[t][i] for i in rng.permutation(len(runs[t]))]
                    need, pieces = int(round(float(length) / dt)) + 1, []
                    for i, rid in enumerate(order):
                        a = arrays[rid]
                        n = len(a["s"])
                        start = int(rng.integers(0, max(n - need, 0) + 1)) if i == 0 else 0
                        stop = min(n, start + need)
                        if stop - start >= 2:
                            pieces.append({k: v[start:stop] for k, v in a.items()})
                            need -= stop - start - 1
                        if need <= 1:
                            break
                    X.append(self._features(pieces, pt))
                    Y.append([np.log(task_params[t][k]) for k in self.free])
        X, Y = np.array(X), np.array(Y)
        g = fg["gbm"]
        self.models = {}
        for i, k in enumerate(self.free):
            m = GradientBoostingRegressor(n_estimators=int(g["n_estimators"]),
                                          max_depth=int(g["max_depth"]),
                                          learning_rate=float(g["learning_rate"]),
                                          subsample=float(g["subsample"]),
                                          random_state=seed + i)
            self.models[k] = m.fit(X, Y[:, i])
        self.info = dict(n_train_tasks=len(tasks), n_samples=len(X), fold_seed=seed,
                         train_r2={k: float(self.models[k].score(X, Y[:, i]))
                                   for i, k in enumerate(self.free)})
        self._cache = {}

    def _predict(self, context):
        key = (context["task"], int(context["draw"]), float(context["length_s"]))
        if key not in self._cache:
            pt = context["task"].split(" | ")[0]
            x = self._features(context["segments"], pt)[None, :]
            params = dict(self.fixed)
            for (lo, hi), k in zip(self.bounds, self.free):
                params[k] = float(np.clip(np.exp(self.models[k].predict(x)[0]), lo, hi))
            self._cache[key] = params
        return self._cache[key]

    def controller(self, context):
        return idm.IDMController(self._predict(context), self.cfg["idm"]["min_gap_m"])

    def context_info(self, context):
        return {f"ctx_idm_{k}": v for k, v in self._predict(context).items()}

    def state(self):
        return dict(method=self.name, n_train_runs=len(self.train_run_ids), **{
            k: v for k, v in self.info.items() if k != "train_r2"},
                    **{f"train_r2_{k}": v for k, v in self.info.get("train_r2", {}).items()})


REGISTRY = {m.name: m for m in (NoAdaptation, NoAdaptationSim, ContextRecalibration,
                                PopulationPrior, M4Pretrained, M4Finetuned, M4FinetunedLOVO,
                                PopulationPriorSamePowertrain, M2bResidual, FeaturesGBM)}
