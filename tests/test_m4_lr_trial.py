import importlib.util
import json
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ema import calibrate, cap_trial, config, data, protocol
from test_context_recalibration import CFG_E, CFG_M, setup
from test_m4_finetune import FT
from test_m4_method import CFG_PRE, save_net


def curve(f, steps=range(0, 3001, 50), skipped=0):
    return [dict(step=s, train_loss=1.0, val_loss=f(s), skipped=skipped) for s in steps]


def test_the_config_file_has_the_agreed_lr_trial():
    lt = config.load_yaml("m4.yaml")["lr_trial"]
    assert lt["folds"] == ["L2_new_vehicle_01", "L2_new_vehicle_09"] and lt["lr"] == 3e-4
    assert lt["max_steps"] == 3000 and lt["early_stopping"] is False
    assert lt["reference"] == "cap_trial" and lt["reference_commit"] == "d386879"
    assert lt["tol"] == 0.01


def test_trial_config_takes_the_lr_of_the_section():
    cfg = {"finetuning": {"max_steps": 1500, "val_every": 50, "patience": 10, "lr": 1e-4},
           "lr_trial": {"lr": 3e-4, "max_steps": 3000, "early_stopping": False}}
    t = cap_trial.trial_config(cfg, "lr_trial")
    assert t["finetuning"]["lr"] == 3e-4 and t["finetuning"]["max_steps"] == 3000
    assert cfg["finetuning"]["lr"] == 1e-4


def test_instability_rule():
    smooth = lambda s: 3.0 + 4.0 * np.exp(-s / 400)
    assert cap_trial.instability(curve(smooth)) is None
    one_jump = curve(lambda s: smooth(s) * (1.15 if s == 2000 else 1.0))
    assert cap_trial.instability(one_jump) is None                      # single spike only
    long_jump = curve(lambda s: smooth(s) * (1.15 if 2000 <= s <= 2100 else 1.0))
    assert "2100" in cap_trial.instability(long_jump)                   # 3 evaluations in a row
    early = curve(lambda s: smooth(s) * (1.5 if 100 <= s <= 300 else 1.0))
    assert cap_trial.instability(early) is None                         # before step 500
    assert cap_trial.instability(curve(lambda s: float("nan") if s == 1000 else smooth(s)))
    assert cap_trial.instability(curve(smooth, skipped=1)) == "skipped steps"


def test_lr_decision():
    ref = curve(lambda s: 3.0 + 4.0 * np.exp(-s / 900), steps=range(0, 4001, 50))
    fast = curve(lambda s: 2.98 + 4.0 * np.exp(-s / 300))                # better and earlier
    adopt, cap, rows = cap_trial.lr_decision([fast, fast], [ref, ref], 3000)
    assert adopt and cap == 1500 and all(r["ok"] for r in rows)
    worse = curve(lambda s: 3.2 + 4.0 * np.exp(-s / 300))
    adopt, cap, rows = cap_trial.lr_decision([fast, worse], [ref, ref], 3000)
    assert not adopt and cap is None and [r["ok"] for r in rows] == [True, False]
    shaky = curve(lambda s: (2.98 + 4.0 * np.exp(-s / 300)) * (1.2 if 2000 <= s <= 2200 else 1))
    adopt, _, rows = cap_trial.lr_decision([fast, shaky], [ref, ref], 3000)
    assert not adopt and rows[1]["unstable"]


def write_trial(directory, history, commit="d386879abc", dirty=False, lr=1e-4, steps=4000):
    cap_trial.save_fold(directory, history, dict(git_commit=commit, git_dirty=dirty,
                                                 finetuning=dict(lr=lr, max_steps=steps)))


def test_check_reference(tmp_path):
    h = curve(lambda s: 3.0, steps=range(0, 4001, 50))
    write_trial(tmp_path / "ok", h)
    assert len(cap_trial.check_reference(tmp_path / "ok", "d386879", 1e-4, 4000)) == len(h)
    for name, kw in [("dirty", dict(dirty=True)), ("commit", dict(commit="aaaaaaa")),
                     ("lr", dict(lr=3e-4)), ("steps", dict(steps=1500))]:
        write_trial(tmp_path / name, h, **kw)
        with pytest.raises(ValueError, match="not the expected trial"):
            cap_trial.check_reference(tmp_path / name, "d386879", 1e-4, 4000)


def test_script_lr_section_writes_its_own_folder_and_decides(tmp_path, monkeypatch, capsys):
    _, cfg_m4 = save_net(tmp_path)
    rt, folds, asg, ct, traj_fd = setup()
    fid = folds[0]["fold_id"]
    cfg_m4 = {**cfg_m4, "finetuning": {**FT, "lr": 1e-3},
              "cap_trial": {"folds": [fid], "max_steps": 6, "early_stopping": False},
              "lr_trial": {"folds": [fid], "lr": 3e-3, "max_steps": 6, "early_stopping": False,
                           "reference": "cap_trial", "reference_commit": "d386879", "tol": 0.01}}
    ref = curve(lambda s: 99.0, steps=range(0, 7, 3))                   # easy to beat
    write_trial(tmp_path / "results" / "m4_cap_trial" / fid, ref, lr=1e-3, steps=6)
    cfgs = {"evaluation.yaml": CFG_E, "models.yaml": CFG_M, "pretraining.yaml": CFG_PRE,
            "m4.yaml": cfg_m4}
    monkeypatch.setattr(config, "load_yaml", lambda n: cfgs[n])
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(data, "load_runs", lambda: None)
    monkeypatch.setattr(data, "load_trajectories", lambda: None)
    monkeypatch.setattr(protocol, "run_table", lambda runs, cfg: rt)
    monkeypatch.setattr(protocol, "make_folds", lambda r, cfg: folds)
    monkeypatch.setattr(protocol, "assign_runs", lambda r, f, cfg: asg)
    monkeypatch.setattr(calibrate, "forward_difference", lambda t, dt: traj_fd)
    monkeypatch.setattr(cap_trial, "git_state", lambda: ("d386879" + "0" * 33, False))
    monkeypatch.setattr(cap_trial, "config_md5", lambda: "x")
    monkeypatch.setattr(sys, "argv", ["m4_cap_trial.py", "--section", "lr_trial"])
    spec = importlib.util.spec_from_file_location(
        "trial_script", config.CONFIG_DIR.parent / "scripts" / "m4_cap_trial.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    script.main()
    out = capsys.readouterr().out
    saved = json.loads((tmp_path / "results" / "m4_lr_trial" / fid / "run_info.json").read_text())
    assert saved["finetuning"]["lr"] == 3e-3 and saved["trial_section"] == "lr_trial"
    assert (tmp_path / "results" / "m4_cap_trial" / fid / "history.csv").exists()   # untouched
    assert "rule (F028): adopt lr 0.003" in out
