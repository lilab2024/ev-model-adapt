import copy

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from ema import evaluate, methods, nets, protocol, train_m4
from test_context_recalibration import CFG_E, CFG_M, DT, setup
from test_m4_method import CFG_PRE, save_net
from test_m4_net import CFG_NET, CFG_TR, SCALES, data

FT = {"lr": 3e-3, "batch_episodes": 4, "targets_per_episode": 2, "target_s": 10.0,
      "context_s": [10, 30], "max_steps": 6, "val_every": 3, "patience": 5,
      "val_episodes_per_task_and_length": 2, "grad_clip": 1.0, "device": "cpu", "seed": 7}


def cfg_m4_with_ft(tmp_path, **changes):
    _, cfg_m4 = save_net(tmp_path)
    return {**cfg_m4, "finetuning": {**FT, **changes}}


def test_cut_pieces_keeps_every_sample_in_order():
    p = {"s": np.arange(25.0), "v_f": np.ones(25), "v_l": np.ones(25)}
    q = {k: v[:11] for k, v in p.items()}
    out = train_m4.cut_pieces([p, q], 10)
    assert [len(c["s"]) for c in out] == [10, 10, 5, 10]     # 1-sample tail of q dropped
    assert np.array_equal(np.concatenate([c["s"] for c in out[:3]]), p["s"])


def test_sampler_with_piece_len_makes_the_same_draws_cut_into_pieces():
    runs, arrays, task_runs, _ = data()
    a = train_m4.EpisodeSampler(arrays, task_runs, CFG_TR, DT, seed=4).sample(10)
    b = train_m4.EpisodeSampler(arrays, task_runs, CFG_TR, DT, seed=4, piece_len=50).sample(10)
    for (ca, ta), (cb, tb) in zip(a, b):
        assert all(len(p["s"]) <= 50 for p in cb)
        expected = train_m4.cut_pieces(ca, 50)              # same draws, then cut
        assert len(cb) == len(expected)
        assert all(np.array_equal(x["s"], y["s"]) for x, y in zip(cb, expected))
        assert all(np.array_equal(x["s"], y["s"]) for x, y in zip(ta, tb))


def test_validation_episodes_take_context_and_targets_from_the_right_runs():
    n = 400
    arrays = {f"r{i}": {"s": i * 1000.0 + np.arange(n), "v_f": np.ones(n), "v_l": np.ones(n)}
              for i in range(6)}
    ctx_runs = {"A": ["r0", "r1"], "B": ["r3"], "C": ["r5"]}
    tgt_runs = {"A": ["r2"], "B": ["r4"]}                     # C has no validation runs
    items = train_m4.validation_episodes(arrays, ctx_runs, tgt_runs, [10, 30], 2, CFG_TR, DT,
                                         seed=1, piece_len=100)
    assert len(items) == 2 * 2 * 2 and {t for t, _, _ in items} == {"A", "B"}
    for task, length, (ctx, targets) in items:
        ids = lambda pieces: {f"r{int(p['s'][0] // 1000)}" for p in pieces}
        assert ids(ctx) <= set(ctx_runs[task]) and ids(targets) <= set(tgt_runs[task])
        assert all(len(p["s"]) <= 100 for p in ctx)
        assert sum(len(p["s"]) - 1 for p in ctx) * DT <= length + 1e-9
    again = train_m4.validation_episodes(arrays, ctx_runs, tgt_runs, [10, 30], 2, CFG_TR, DT,
                                         seed=1, piece_len=100)
    assert all(np.array_equal(x[2][0][0]["s"], y[2][0][0]["s"]) for x, y in zip(items, again))
    with pytest.raises(ValueError, match="overlap"):
        train_m4.validation_episodes(arrays, {"A": ["r0"]}, {"A": ["r0"]}, [10], 1, CFG_TR, DT, 1)


def test_eval_at_start_keeps_the_starting_weights_without_improvement():
    runs, arrays, task_runs, _ = data()
    torch.manual_seed(0)
    net = nets.AdaptNet(CFG_NET, {"v0": [25.0, 60.0], "T": [0.3, 4.0], "s0": [0.5, 15.0],
                                  "a": [0.2, 4.0], "b": [0.01, 6.0], "delta": 4.0})
    start = copy.deepcopy(net.state_dict())
    val = train_m4.EpisodeSampler(arrays, task_runs, CFG_TR, DT, seed=99).sample(8)
    sampler = train_m4.EpisodeSampler(arrays, task_runs, CFG_TR, DT, seed=2)
    frozen = {**CFG_TR, "lr": 0.0, "steps": 6, "val_every": 3}
    hist = train_m4.train(net, sampler, val, frozen, DT, SCALES, log=lambda *a: None,
                          eval_at_start=True)
    assert hist[0]["step"] == 0 and np.isnan(hist[0]["train_loss"])
    assert all(torch.equal(start[k], v) for k, v in net.state_dict().items())
    hist2 = train_m4.train(net, sampler, val, CFG_TR, DT, SCALES, log=lambda *a: None,
                           eval_at_start=True)
    assert min(h["val_loss"] for h in hist2) <= hist2[0]["val_loss"]


def test_m4_finetuned_trains_per_fold_saves_artifacts_and_is_reproducible(tmp_path):
    cfg_m4 = cfg_m4_with_ft(tmp_path)
    rt, folds, asg, ct, traj_fd = setup()
    cls = lambda cfg_m, cfg_e: methods.M4Finetuned(cfg_m, cfg_e, cfg_m4, CFG_PRE,
                                                    log=lambda *a: None)
    res, st = evaluate.run_evaluation(cls, CFG_M, rt, folds, asg, ct, traj_fd, CFG_E,
                                      verbose=False, artifact_dir=tmp_path / "out")
    assert {"ctx_idm_T", "ctx_n_pieces"} <= set(res.columns) and not res["collided"].any()
    for s, f in zip(st, folds):
        assert s["method"] == "m4a_finetuned" and s["n_train_runs"] > 0 and s["n_val_runs"] > 0
        assert s["val_loss_best"] <= s["val_loss_start"] and s["steps_run"] <= FT["max_steps"]
        assert s["device"] == "cpu"
        train_ids = set(asg[(asg["fold_id"] == f["fold_id"]) & (asg["role"] == "train")]["run_id"])
        val_ids = set(asg[(asg["fold_id"] == f["fold_id"]) & (asg["role"] == "val")]["run_id"])
        assert s["n_train_runs"] == len(train_ids) and s["n_val_runs"] == len(val_ids)
        saved = torch.load(tmp_path / "out" / "nets" / f"{f['fold_id']}.pt")
        hist = pd.read_csv(tmp_path / "out" / "histories" / f"{f['fold_id']}.csv")
        assert hist["step"].iloc[0] == 0 and set(saved) >= {"log_lo", "log_hi"}
    res2, st2 = evaluate.run_evaluation(cls, CFG_M, rt, folds, asg, ct, traj_fd, CFG_E,
                                        verbose=False)
    pd.testing.assert_frame_equal(res, res2)
    assert [s["val_loss_best"] for s in st] == [s["val_loss_best"] for s in st2]


def test_m4_finetuned_without_learning_equals_the_pretrained_method(tmp_path):
    cfg_m4 = cfg_m4_with_ft(tmp_path, lr=0.0)
    rt, folds, asg, ct, traj_fd = setup()
    methods.M4Pretrained._cache.clear()
    pre, _ = evaluate.run_evaluation(
        lambda cfg_m, cfg_e: methods.M4Pretrained(cfg_m, cfg_e, cfg_m4, CFG_PRE),
        CFG_M, rt, folds, asg, ct, traj_fd, CFG_E, verbose=False)
    ft, st = evaluate.run_evaluation(
        lambda cfg_m, cfg_e: methods.M4Finetuned(cfg_m, cfg_e, cfg_m4, CFG_PRE,
                                                 log=lambda *a: None),
        CFG_M, rt, folds, asg, ct, traj_fd, CFG_E, verbose=False)
    cols = ["fold_id", "task", "draw", "length_s", "run_id", "spacing_rmse", "ctx_idm_T"]
    pd.testing.assert_frame_equal(pre[cols], ft[cols])
    assert all(s["best_step"] == 0 for s in st)
