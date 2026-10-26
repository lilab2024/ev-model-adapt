import copy

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from ema import calibrate, context, evaluate, methods, nets, protocol, simulate, train_m4
from ema.models import idm
from test_context_recalibration import CFG_E, CFG_M, DT, P_EV, P_ICE
from test_m4_finetune import FT
from test_m4_method import CFG_PRE, save_net
from test_m4_net import CFG_NET, CFG_TR, SCALES, data

LEVELS = ["L2_new_vehicle", "L2_no_ice", "L3_cross_powertrain"]


def four_vehicle_setup(levels=LEVELS):
    """EV A and B, ICE C and D (two gaps each, 4 configurations x 1 run of 60 s)."""
    runs, trajs = [], []
    t = np.arange(600) * DT
    specs = [("EV", "A", P_EV, 1.0), ("EV", "B", P_EV, 1.2), ("ICE", "C", P_ICE, 1.0),
             ("ICE", "D", P_ICE, 0.8)]
    for pt, veh, p, scale in specs:
        p = dict(p, T=p["T"] * scale)
        for gap in ["g1", "g2"]:
            for k in range(4):
                rid = f"{pt}_{veh}_{gap}_{k}"
                v_lead = 20.0 - 6.0 * (1 + 0.2 * k) * np.exp(-((t - 25.0) / 6.0) ** 2)
                sim = simulate.rollout(idm.IDMController(p), v_lead, s0=40.0, v0=20.0, dt=DT)
                trajs.append(pd.DataFrame({"run_id": rid, "t": t, "v_f": sim["v"],
                                           "v_l": v_lead, "s": sim["s"]}))
                runs.append(dict(run_id=rid, powertrain=pt, vehicle=veh, gap=gap, profile="dip",
                                 desired_mph=0.0, v_free_mph=45.0, v_reduced_mph=20.0 + 5 * k,
                                 qc_pass=True, exclude=False, duration_s=float(t[-1])))
    cfg_e = dict(CFG_E, levels=levels)
    rt = protocol.run_table(pd.DataFrame(runs), cfg_e)
    folds = protocol.make_folds(rt, cfg_e)
    asg = protocol.assign_runs(rt, folds, cfg_e)
    ct = context.context_table(rt, cfg_e)
    return cfg_e, rt, folds, asg, ct, calibrate.forward_difference(pd.concat(trajs), DT)


def make(tmp_path, cfg_e, **ft):
    _, cfg_m4 = save_net(tmp_path)
    cfg_m4 = {**cfg_m4, "finetuning": {**FT, **ft}}
    return lambda cfg_m, c: methods.M4FinetunedLOVO(cfg_m, cfg_e, cfg_m4, CFG_PRE,
                                                    log=lambda *a: None)


def test_train_fixed_equals_train_after_the_same_number_of_steps():
    runs, arrays, task_runs, _ = data()
    val = train_m4.EpisodeSampler(arrays, task_runs, CFG_TR, DT, seed=99).sample(4)
    nets_ = []
    for fixed in (True, False):
        torch.manual_seed(0)
        net = nets.AdaptNet(CFG_NET, {"v0": [25.0, 60.0], "T": [0.3, 4.0], "s0": [0.5, 15.0],
                                      "a": [0.2, 4.0], "b": [0.01, 6.0], "delta": 4.0})
        sampler = train_m4.EpisodeSampler(arrays, task_runs, CFG_TR, DT, seed=2)
        cfg = {**CFG_TR, "steps": 5, "val_every": 5, "patience": 1}
        if fixed:
            hist = train_m4.train_fixed(net, sampler, cfg, DT, SCALES, log=lambda *a: None)
            assert [h["step"] for h in hist] == [5]
        else:            # one validation at the last step: train() keeps exactly those weights
            train_m4.train(net, sampler, val, cfg, DT, SCALES, log=lambda *a: None)
        nets_.append(net.state_dict())
    assert all(torch.equal(nets_[0][k], nets_[1][k]) for k in nets_[0])


def test_held_out_vehicle_rule_and_two_stages(tmp_path):
    cfg_e, rt, folds, asg, ct, traj_fd = four_vehicle_setup()
    cls = make(tmp_path, cfg_e)
    res, st = evaluate.run_evaluation(cls, CFG_M, rt, folds, asg, ct, traj_fd, cfg_e,
                                      verbose=False, artifact_dir=tmp_path / "out")
    by = {s["fold_id"]: s for s in st}
    test_vehicle = {f["fold_id"]: f["test"][0].split(" | ")[1] for f in folds}
    for f in folds:
        s = by[f["fold_id"]]
        train_vehicles = sorted({t.split(" | ")[1] for t in f["train"]})
        if len(train_vehicles) < 2:                  # L2_no_ice: one EV left, no hold-out
            assert f["level"] == "L2_no_ice" and not s["lovo_used"]
            continue
        assert s["lovo_used"] and s["heldout_vehicle"] in train_vehicles
        assert s["heldout_vehicle"] != test_vehicle[f["fold_id"]]
        if f["level"] == "L3_cross_powertrain":      # no EV in training: other powertrain
            assert s["heldout_vehicle"] in ("C", "D") and not s["heldout_same_powertrain"]
        else:                                        # L2: the other vehicle of the same powertrain
            same = {"A": "B", "B": "A", "C": "D", "D": "C"}[test_vehicle[f["fold_id"]]]
            assert s["heldout_vehicle"] == same and s["heldout_same_powertrain"]
        assert s["stage2_steps"] == s["best_step"] <= FT["max_steps"]
        assert s["n_train_tasks_stage1"] == s["n_train_tasks_sampled"] - 2
        assert all(t.split(" | ")[1] == s["heldout_vehicle"] for t in s["heldout_tasks"])
        if s["stage2_steps"] > 0:
            h2 = pd.read_csv(tmp_path / "out" / "histories" / f"{f['fold_id']}_stage2.csv")
            assert h2["step"].iloc[-1] == s["stage2_steps"]
    assert {"ctx_idm_T"} <= set(res.columns) and not res["collided"].any()
    res2, _ = evaluate.run_evaluation(cls, CFG_M, rt, folds, asg, ct, traj_fd, cfg_e, verbose=False)
    pd.testing.assert_frame_equal(res, res2)                 # reproducible on CPU


def test_stage_two_uses_all_vehicles_and_the_original_seeds(tmp_path, monkeypatch):
    """Stage 2 must be m4a_finetuned's training (all training vehicles, same seeds) stopped
    after S steps; S is forced to 6 by replacing the stage-1 training."""
    cfg_e, rt, folds, asg, ct, traj_fd = four_vehicle_setup(["L2_new_vehicle"])
    f = folds[0]
    ids = sorted(asg.loc[(asg["fold_id"] == f["fold_id"]) & (asg["role"] == "train"), "run_id"])
    fake = [dict(step=0, train_loss=float("nan"), val_loss=5.0, elapsed_s=0.0, skipped=0),
            dict(step=6, train_loss=1.0, val_loss=4.0, elapsed_s=0.0, skipped=0)]
    original_train = train_m4.train
    monkeypatch.setattr(train_m4, "train", lambda *a, **k: fake)
    lovo = make(tmp_path, cfg_e)(CFG_M, cfg_e)
    lovo.set_fold(f)
    lovo.fit(traj_fd, rt, ids)
    monkeypatch.setattr(train_m4, "train", original_train)
    assert lovo.info["stage2_steps"] == 6 and lovo.history2[-1]["step"] == 6

    # the same training as m4a_finetuned would do (all training vehicles, its fold seed)
    tasks = sorted(rt.loc[ids, "task"].unique())
    fold_seed = int(np.random.SeedSequence(methods._seed(FT["seed"], *tasks)).generate_state(1)[0])
    cfg_tr = dict(context_s=FT["context_s"], target_s=FT["target_s"],
                  targets_per_episode=FT["targets_per_episode"], batch_episodes=FT["batch_episodes"],
                  steps=6, lr=FT["lr"], grad_clip=FT["grad_clip"], val_every=FT["val_every"],
                  patience=FT["patience"], seed=fold_seed)
    arrays = train_m4.run_arrays(traj_fd, ids)
    runs = {t: sorted(rt.loc[ids].index[rt.loc[ids, "task"] == t]) for t in tasks}
    sampler = train_m4.EpisodeSampler(arrays, runs, cfg_tr, DT, fold_seed, piece_len=60)
    net, _ = methods.load_pretrained_net(make_cfg(tmp_path), CFG_PRE, CFG_M)
    train_m4.train_fixed(net, sampler, cfg_tr, DT, CFG_NET["scales"], log=lambda *a: None)
    a, b = lovo.net.state_dict(), net.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


def make_cfg(tmp_path):
    _, cfg_m4 = save_net(tmp_path)
    return cfg_m4


def test_lovo_refuses_level_one(tmp_path):
    cfg_e, rt, folds, asg, ct, traj_fd = four_vehicle_setup(["L1_new_setting"])
    with pytest.raises(ValueError, match="only at the levels"):
        evaluate.run_evaluation(make(tmp_path, cfg_e), CFG_M, rt, folds[:1], asg, ct, traj_fd,
                                cfg_e, verbose=False)
