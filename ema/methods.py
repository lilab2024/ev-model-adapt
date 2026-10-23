"""Adaptation methods. Every method follows the same interface:

    method = Method(cfg_models)
    method.fit(traj_fd, rt, train_run_ids)   # learn from the training runs of a fold only
    method.uses_context                      # False: the context is ignored
    controller = method.controller(context)  # controller for one held-out task and context
    method.state()                           # dict saved with the results (e.g. parameters)
"""
from ema import calibrate, config
from ema.models import idm


class NoAdaptation:
    """M0: one IDM parameter set calibrated on the pooled training runs; no adaptation."""

    name = "no_adaptation"
    uses_context = False

    def __init__(self, cfg_models=None):
        self.cfg = cfg_models or config.load_yaml("models.yaml")
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


REGISTRY = {NoAdaptation.name: NoAdaptation}
