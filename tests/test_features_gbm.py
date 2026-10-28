import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from ema import config, context_features, evaluate, methods, simulate
from ema.models import idm
from test_population_prior import CFG_E, CFG_M, DT, setup

CFG_FG = {"features": {"tg_min_speed": 5.0, "stop_speed": 0.5}, "context_s": [10, 40],
          "contexts_per_task_length": 3,
          "gbm": {"n_estimators": 20, "max_depth": 2, "learning_rate": 0.1, "subsample": 0.8},
          "seed": 5}


def test_the_config_file_has_the_agreed_settings():
    c = config.load_yaml("features_gbm.yaml")
    assert c["context_s"] == [10, 30, 60, 120, 300] and c["contexts_per_task_length"] == 40
    assert c["gbm"] == {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8}
    assert c["features"] == {"tg_min_speed": 5.0, "stop_speed": 0.5}


def test_features_are_finite_and_meaningful():
    t = np.arange(300) * DT
    v_lead = 20.0 - 5.0 * np.exp(-((t - 15.0) / 4.0) ** 2)
    p = dict(v0=33.0, T=1.5, s0=2.5, a=1.0, b=1.5, delta=4.0)
    sim = simulate.rollout(idm.IDMController(p), v_lead, s0=35.0, v0=20.0, dt=DT)
    piece = pd.DataFrame({"s": sim["s"], "v_f": sim["v"], "v_l": v_lead})
    f = dict(zip(context_features.FEATURE_NAMES, context_features.context_features([piece], DT, True)))
    assert len(f) == len(context_features.FEATURE_NAMES) and all(np.isfinite(list(f.values())))
    assert f["is_ev"] == 1.0 and f["length_s"] == pytest.approx(299 * DT)
    assert 1.0 < f["tg_median"] < 2.5 and f["stop_share"] == 0.0
    still = {"s": np.full(50, 10.0), "v_f": np.zeros(50), "v_l": np.zeros(50)}    # standing still
    g = context_features.context_features([still], DT, False)
    assert np.all(np.isfinite(g)) and g[-1] == 0.0


def make():
    return lambda cfg_m, cfg_e: methods.FeaturesGBM(cfg_m, cfg_e, CFG_FG)


def test_features_gbm_runs_and_is_reproducible():
    methods.PopulationPrior._task_fits.clear()
    rt, folds, asg, ct, traj_fd = setup()
    res, st = evaluate.run_evaluation(make(), CFG_M, rt, folds, asg, ct, traj_fd, CFG_E,
                                      verbose=False)
    for f, s in zip(folds, st):
        n_tasks = len(f["train"])
        assert s["method"] == "features_gbm" and s["n_train_tasks"] == n_tasks
        assert s["n_samples"] == n_tasks * len(CFG_FG["context_s"]) * CFG_FG["contexts_per_task_length"]
    for k, spec in CFG_M["idm"]["params"].items():
        if isinstance(spec, (list, tuple)):
            assert res[f"ctx_idm_{k}"].between(spec[0], spec[1]).all()
    assert not res["collided"].any()
    res2, _ = evaluate.run_evaluation(make(), CFG_M, rt, folds, asg, ct, traj_fd, CFG_E,
                                      verbose=False)
    pd.testing.assert_frame_equal(res, res2)
