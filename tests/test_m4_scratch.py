import copy

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from ema import evaluate, methods
from test_m4_lovo import CFG_M, CFG_PRE, FT, four_vehicle_setup, save_net


def cls(tmp_path, cfg_e, kind=methods.M4ScratchLOVO):
    _, cfg_m4 = save_net(tmp_path)
    cfg_m4 = {**cfg_m4, "finetuning": {**FT}}
    return lambda cfg_m, c: kind(cfg_m, cfg_e, cfg_m4, CFG_PRE, log=lambda *a: None), cfg_m4


def test_scratch_starts_from_a_reproducible_random_network(tmp_path):
    cfg_e, rt, folds, asg, ct, traj_fd = four_vehicle_setup(["L2_new_vehicle"])
    make, cfg_m4 = cls(tmp_path, cfg_e)
    m = make(CFG_M, cfg_e)
    pretrained = {k: v.clone() for k, v in m.net.state_dict().items()}
    a, b = m._initial_net(7).state_dict(), m._initial_net(7).state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not all(torch.equal(a[k], pretrained[k]) for k in a if k not in ("log_lo", "log_hi"))
    res, st = evaluate.run_evaluation(make, CFG_M, rt, folds, asg, ct, traj_fd, cfg_e, verbose=False)
    assert all(s["method"] == "m4a_scratch_lovo" and s["init"] == "random" for s in st)
    assert all(s["weights_md5"] == "none (random initialisation)" for s in st)
    res2, _ = evaluate.run_evaluation(make, CFG_M, rt, folds, asg, ct, traj_fd, cfg_e, verbose=False)
    pd.testing.assert_frame_equal(res, res2)


def test_scratch_with_the_pretrained_start_equals_lovo(tmp_path, monkeypatch):
    cfg_e, rt, folds, asg, ct, traj_fd = four_vehicle_setup()
    make, cfg_m4 = cls(tmp_path, cfg_e)
    pre = methods.M4Pretrained(CFG_M, cfg_e, cfg_m4, CFG_PRE).net
    monkeypatch.setattr(methods.M4ScratchLOVO, "_initial_net", lambda self, seed: copy.deepcopy(pre))
    scratch, _ = evaluate.run_evaluation(make, CFG_M, rt, folds, asg, ct, traj_fd, cfg_e, verbose=False)
    lovo_make, _ = cls(tmp_path, cfg_e, methods.M4FinetunedLOVO)
    lovo, _ = evaluate.run_evaluation(lovo_make, CFG_M, rt, folds, asg, ct, traj_fd, cfg_e,
                                      verbose=False)
    assert set(scratch["method"]) == {"m4a_scratch_lovo"}
    pd.testing.assert_frame_equal(scratch.drop(columns="method"), lovo.drop(columns="method"))
