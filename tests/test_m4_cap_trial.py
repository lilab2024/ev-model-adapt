import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from ema import cap_trial, config
from test_context_recalibration import CFG_E, CFG_M, setup
from test_m4_finetune import FT
from test_m4_method import CFG_PRE, save_net


def test_the_config_file_has_the_agreed_trial():
    ct = config.load_yaml("m4.yaml")["cap_trial"]
    assert ct["folds"] == ["L2_new_vehicle_01", "L2_new_vehicle_09"]
    assert ct["max_steps"] == 4000 and ct["early_stopping"] is False


def test_trial_config_sets_the_cap_and_switches_off_early_stopping():
    cfg = {"finetuning": {"max_steps": 1500, "val_every": 50, "patience": 10, "lr": 1e-4},
           "cap_trial": {"max_steps": 4000, "early_stopping": False}}
    t = cap_trial.trial_config(cfg)
    assert t["finetuning"]["max_steps"] == 4000 and t["finetuning"]["patience"] > 4000 // 50
    assert t["finetuning"]["lr"] == 1e-4 and cfg["finetuning"]["max_steps"] == 1500
    cfg["cap_trial"]["early_stopping"] = True
    assert cap_trial.trial_config(cfg)["finetuning"]["patience"] == 10


def test_choose_cap_follows_the_rule():
    steps = np.arange(0, 4001, 50)
    flat = [dict(step=s, val_loss=3.0 + 4.0 * np.exp(-s / 300)) for s in steps]   # flat early
    late = [dict(step=s, val_loss=3.0 + 4.0 * np.exp(-s / 700)) for s in steps]   # flat later
    still = [dict(step=s, val_loss=7.0 - s / 1000) for s in steps]                 # never flat
    assert cap_trial.choose_cap([flat], 4000) == 1500
    assert cap_trial.choose_cap([flat, late], 4000) == 3500
    assert cap_trial.choose_cap([flat, still], 4000) is None
    bump = flat + [dict(step=4050, val_loss=9.0)]                                  # overfitting
    assert cap_trial.choose_cap([bump], 4050) == 1500


def test_run_fold_trains_without_early_stopping_and_saves(tmp_path):
    _, cfg_m4 = save_net(tmp_path)
    cfg_m4 = {**cfg_m4, "finetuning": {**FT, "lr": 0.0, "patience": 1},
              "cap_trial": {"folds": [], "max_steps": 6, "early_stopping": False}}
    rt, folds, asg, ct, traj_fd = setup()
    fid = folds[0]["fold_id"]
    trial = cap_trial.trial_config(cfg_m4)
    hist, info, train_ids = cap_trial.run_fold(fid, rt, asg, traj_fd, CFG_M, CFG_E, trial,
                                               CFG_PRE, log=lambda *a: None)
    a = asg[asg["fold_id"] == fid]
    assert sorted(train_ids) == sorted(a.loc[a["role"] == "train", "run_id"])
    assert info["n_val_runs"] == (a["role"] == "val").sum()
    assert hist[-1]["step"] == 6                     # with patience 1 it would stop at step 3
    cap_trial.save_fold(tmp_path / fid, hist, dict(fold_id=fid, **info))
    assert json.loads((tmp_path / fid / "run_info.json").read_text())["fold_id"] == fid
    assert cap_trial.load_history(tmp_path / fid)[-1]["step"] == 6
