import hashlib
import importlib.util

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from ema import config, evaluate, methods, nets, sim_calibrate, train_m4
from test_context_recalibration import CFG_E, CFG_M, DT, setup
from test_m4_net import CFG_NET

CFG_PRE = {"network": CFG_NET}


def save_net(tmp_path, spec=None, seed=0):
    torch.manual_seed(seed)
    net = nets.AdaptNet(CFG_NET, spec or CFG_M["idm"]["params"])
    path = tmp_path / f"net{seed}.pt"
    torch.save(net.state_dict(), path)
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    cfg_m4 = {"pretrained": {"weights": str(path), "md5": md5}, "context_piece_s": 6.0,
              "eval_device": "cpu"}
    return net, cfg_m4


def make_method(cfg_m4):
    return methods.M4Pretrained(CFG_M, CFG_E, cfg_m4, CFG_PRE)


def piece(n, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"s": 30 + rng.normal(size=n).cumsum(), "v_f": 20 + rng.normal(size=n),
                         "v_l": 20 + rng.normal(size=n)})


def test_the_config_file_names_the_r3_weights():
    c = config.load_yaml("m4.yaml")
    assert c["pretrained"]["weights"].endswith("m4_pretrain_r3/net.pt")
    assert len(c["pretrained"]["md5"]) == 32 and int(c["pretrained"]["md5"], 16) >= 0
    assert c["context_piece_s"] == 60.0


def test_wrong_md5_or_other_bounds_are_refused(tmp_path):
    _, cfg_m4 = save_net(tmp_path)
    bad = {**cfg_m4, "pretrained": {**cfg_m4["pretrained"], "md5": "0" * 32}}
    with pytest.raises(ValueError, match="md5"):
        make_method(bad)
    other = {**CFG_M["idm"]["params"], "b": [0.01, 6.0]}
    _, cfg_other = save_net(tmp_path, spec=other, seed=1)
    with pytest.raises(ValueError, match="bounds"):
        make_method(cfg_other)


def test_contexts_are_cut_into_pieces_and_predictions_cached(tmp_path):
    methods.M4Pretrained._cache.clear()
    net, cfg_m4 = save_net(tmp_path)
    m = make_method(cfg_m4)
    segs = [piece(150, 0), piece(40, 1)]                     # 14.9 s and 3.9 s
    pieces = m.context_pieces(segs)
    assert [len(p) for p in pieces] == [60, 60, 30, 40]      # at most 6 s = 60 samples
    ctx = dict(task="EV | A | S", draw=0, length_s=18.0, segments=segs)
    ctrl = m.controller(ctx)
    expected = train_m4.predict_params(net, pieces, DT, CFG_NET["scales"])
    assert ctrl.p == pytest.approx(expected)
    whole = train_m4.predict_params(net, segs, DT, CFG_NET["scales"])
    assert ctrl.p != pytest.approx(whole)                   # cutting changes the input
    assert m.context_info(ctx)["ctx_n_pieces"] == 4
    m2 = make_method(cfg_m4)
    m2.controller(ctx)
    assert (m.n_new, m2.n_new, m2.n_cached) == (1, 0, 1)


def test_m4_pretrained_runs_through_the_evaluation(tmp_path):
    methods.M4Pretrained._cache.clear()
    _, cfg_m4 = save_net(tmp_path)
    rt, folds, asg, ct, traj_fd = setup()
    cls = lambda cfg_m, cfg_e: methods.M4Pretrained(cfg_m, cfg_e, cfg_m4, CFG_PRE)
    res, st = evaluate.run_evaluation(cls, CFG_M, rt, folds, asg, ct, traj_fd, CFG_E,
                                      verbose=False)
    n_ctx = len(ct[ct["role"] == "context"].groupby(["task", "draw", "length_s"]))
    assert set(res["length_s"]) == {10.0, 40.0}
    assert {"ctx_idm_T", "ctx_n_pieces"} <= set(res.columns)
    assert (res["ctx_idm_T"] >= 0.3 - 1e-6).all() and (res["ctx_idm_T"] <= 4.0 + 1e-6).all()
    assert all(s["n_train_runs"] == 0 for s in st)
    assert sum(s["n_new_predictions"] for s in st) == n_ctx
    _, st2 = evaluate.run_evaluation(cls, CFG_M, rt, folds, asg, ct, traj_fd, CFG_E,
                                     verbose=False)
    assert sum(s["n_new_predictions"] for s in st2) == 0


def test_fold_reuse_hash_covers_the_m4_config(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("evaluate_script",
                                                  config.REPO_ROOT / "scripts" / "evaluate.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    for name in ["data.yaml", "evaluation.yaml", "models.yaml", "pretraining.yaml", "m4.yaml"]:
        (tmp_path / name).write_bytes((config.CONFIG_DIR / name).read_bytes())
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    before = script.config_md5()
    for name in ["pretraining.yaml", "m4.yaml"]:
        original = (tmp_path / name).read_text()
        (tmp_path / name).write_text(original + "# changed")
        assert script.config_md5() != before
        (tmp_path / name).write_text(original)
    assert script.config_md5() == before
