import importlib.util
import sys

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from ema import calibrate, config, context, data, evaluate, methods, protocol
from test_context_recalibration import CFG_E, CFG_M, setup
from test_m4_method import CFG_PRE, save_net


def test_shards_are_disjoint_and_complete():
    folds = [{"fold_id": f"f{i}"} for i in range(29)]
    parts = [evaluate.shard(folds, f"{k}/4") for k in range(4)]
    ids = [f["fold_id"] for p in parts for f in p]
    assert sorted(ids) == sorted(f["fold_id"] for f in folds) and len(ids) == len(set(ids))
    assert [f["fold_id"] for f in parts[1]][:3] == ["f1", "f5", "f9"]
    assert evaluate.shard(folds, "0/1") == folds
    for bad in ["4/4", "-1/4", "1/0", "x", "1-4"]:
        with pytest.raises(ValueError):
            evaluate.shard(folds, bad)


def test_the_config_file_has_the_d022_settings():
    c = config.load_yaml("m4.yaml")
    assert c["finetuning"]["lr"] == 3e-4 and c["finetuning"]["max_steps"] == 3000
    assert c["finetuning"]["patience"] == 10 and c["cap_trial"]["lr"] == 1e-4


def run_script(argv, root, monkeypatch, cfg_m4):
    rt, folds, asg, ct, traj_fd = setup()
    cfgs = {"evaluation.yaml": CFG_E, "models.yaml": CFG_M, "pretraining.yaml": CFG_PRE,
            "m4.yaml": cfg_m4, "data.yaml": {"processed": {"md5": {"trajectories": "x"}}}}
    monkeypatch.setattr(config, "load_yaml", lambda n: cfgs[n])
    monkeypatch.setattr(config, "REPO_ROOT", root)
    monkeypatch.setattr(data, "load_runs", lambda: None)
    monkeypatch.setattr(data, "load_trajectories", lambda: None)
    monkeypatch.setattr(protocol, "run_table", lambda runs, cfg: rt)
    monkeypatch.setattr(protocol, "make_folds", lambda r, cfg: folds)
    monkeypatch.setattr(protocol, "assign_runs", lambda r, f, cfg: asg[asg["fold_id"].isin(
        [x["fold_id"] for x in f])])
    monkeypatch.setattr(context, "context_table", lambda r, cfg: ct)
    monkeypatch.setattr(calibrate, "forward_difference", lambda t, dt: traj_fd)
    monkeypatch.setattr(sys, "argv", ["evaluate.py"] + argv)
    spec = importlib.util.spec_from_file_location(
        "evaluate_script", config.CONFIG_DIR.parent / "scripts" / "evaluate.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(script, "git_state", lambda: ("a" * 40, False))
    monkeypatch.setattr(script, "config_md5", lambda: "same")
    script.main()
    return folds


def test_sharded_runs_then_summary_equal_one_run(tmp_path, monkeypatch, capsys):
    _, cfg_m4 = save_net(tmp_path)
    m = ["--method", "m4a_pretrained"]
    methods.M4Pretrained._cache.clear()
    folds = run_script(m, tmp_path / "single", monkeypatch, cfg_m4)
    assert len(folds) >= 2
    methods.M4Pretrained._cache.clear()
    for k in range(2):
        run_script(m + ["--shard", f"{k}/2", "--no-summary"], tmp_path / "par", monkeypatch, cfg_m4)
        assert not (tmp_path / "par" / "results" / "m4a_pretrained" / "runs.csv").exists()
    capsys.readouterr()
    run_script(m, tmp_path / "par", monkeypatch, cfg_m4)
    assert f"({len(folds)} reused, 0 to run)" in capsys.readouterr().out
    one, par = (pd.read_csv(tmp_path / d / "results" / "m4a_pretrained" / "runs.csv")
                for d in ["single", "par"])
    pd.testing.assert_frame_equal(one, par)
    with pytest.raises(SystemExit):                          # --shard without --no-summary
        run_script(m + ["--shard", "0/2"], tmp_path / "bad", monkeypatch, cfg_m4)
