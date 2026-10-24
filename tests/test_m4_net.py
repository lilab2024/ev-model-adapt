import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ema import nets, synthetic, train_m4

DT = 0.1
CFG_M = {"idm": {"params": {"v0": [25.0, 60.0], "T": [0.3, 4.0], "s0": [0.5, 15.0],
                            "a": [0.2, 4.0], "b": [0.01, 6.0], "delta": 4.0},
                 "min_gap_m": 0.1}}
CFG_P = {"synthetic": {
    "n_vehicles": 40, "runs_per_vehicle": 3, "run_s": 60.0,
    "leader": {"base_speed_mps": [12.0, 30.0], "kinds": ["dip", "osc", "step"],
               "dip_depth_frac": [0.2, 1.0], "dip_start_s": [5.0, 20.0], "hold_s": [0.0, 15.0],
               "dip_stop_share": 0.4, "osc_amplitude_mps": [1.0, 5.0],
               "osc_period_s": [10.0, 40.0], "step_size_mps": [3.0, 8.0],
               "step_every_s": [10.0, 25.0], "decel_mps2": [0.5, 2.5], "accel_mps2": [0.5, 2.0]},
    "init_gap_factor": [0.8, 1.3], "noise": {"spacing_m": 0.2, "speed_mps": 0.05},
    "smoothing": {"window_s": 1.0, "polyorder": 2},
    "reject": {"min_gap_m": 0.5, "max_gap_m": 150.0}, "seed": 3}}
CFG_NET = {"scales": {"s": 50.0, "v_f": 30.0, "dv": 5.0, "a_fd": 2.0}, "encoder": "gru",
           "hidden": 16, "z_dim": 4, "head_hidden": 16}
CFG_TR = {"context_s": [10, 30], "target_s": 30.0, "targets_per_episode": 2,
          "batch_episodes": 8, "steps": 40, "lr": 3e-3, "grad_clip": 1.0,
          "val_every": 10, "patience": 100, "seed": 0}
SCALES = CFG_NET["scales"]


def data():
    runs, traj, params = synthetic.make_dataset(CFG_P, CFG_M, DT)
    arrays = train_m4.run_arrays(traj)
    task_runs = runs.groupby("task").groups
    return runs, arrays, {t: list(r) for t, r in task_runs.items()}, params


def test_features_are_scaled_and_have_the_right_shape():
    f = nets.piece_features(np.array([50.0, 50.0, 50.0]), np.array([30.0, 30.2, 30.4]),
                            np.array([35.0, 35.0, 35.0]), DT, SCALES)
    assert f.shape == (3, 4) and f.dtype == np.float32
    assert np.allclose(f[:, 0], 1.0) and np.allclose(f[:, 2], [5 / 5, 4.8 / 5, 4.6 / 5])
    assert np.allclose(f[:, 3], 2.0 / 2.0)


def test_head_outputs_stay_within_bounds():
    torch.manual_seed(0)
    net = nets.AdaptNet(CFG_NET, CFG_M["idm"]["params"])
    p = net.params_from_z(torch.randn(50, CFG_NET["z_dim"]) * 100)
    for k, (lo, hi) in [(k, v) for k, v in CFG_M["idm"]["params"].items() if isinstance(v, list)]:
        assert (p[k] >= lo * (1 - 1e-5)).all() and (p[k] <= hi * (1 + 1e-5)).all()
    assert torch.allclose(p["delta"], torch.tensor(4.0))


def test_encoder_ignores_padding_and_piece_order():
    torch.manual_seed(0)
    net = nets.AdaptNet(CFG_NET, CFG_M["idm"]["params"]).eval()
    rng = np.random.default_rng(0)
    a = rng.normal(size=(30, 4)).astype(np.float32)
    b = rng.normal(size=(80, 4)).astype(np.float32)
    with torch.no_grad():
        z1 = net.encode(*nets.collate_contexts([[a, b]]), 1)
        z2 = net.encode(*nets.collate_contexts([[b, a]]), 1)
        z3 = net.encode(*nets.collate_contexts([[a]]), 1)
        z4 = net.encode(*nets.collate_contexts([[a], [b, b]]), 2)[0:1]
    assert torch.allclose(z1, z2, atol=1e-6)
    assert torch.allclose(z3, z4, atol=1e-6)


def test_episodes_use_disjoint_runs_of_one_task():
    runs, arrays, task_runs, _ = data()
    s = train_m4.EpisodeSampler(arrays, task_runs, CFG_TR, DT, seed=1)
    for ctx, targets in s.sample(20):
        assert ctx and targets
        assert sum(len(p["s"]) - 1 for p in ctx) * DT <= 30.0 + 1e-9
        assert all(len(t["s"]) <= 301 for t in targets)
    s1 = train_m4.EpisodeSampler(arrays, task_runs, CFG_TR, DT, seed=1).sample(5)
    s2 = train_m4.EpisodeSampler(arrays, task_runs, CFG_TR, DT, seed=1).sample(5)
    assert all(np.array_equal(a[0][0]["s"], b[0][0]["s"]) for a, b in zip(s1, s2))


def test_training_reduces_the_closed_loop_loss():
    runs, arrays, task_runs, _ = data()
    torch.manual_seed(0)
    net = nets.AdaptNet(CFG_NET, CFG_M["idm"]["params"])
    val = train_m4.EpisodeSampler(arrays, task_runs, CFG_TR, DT, seed=99).sample(16)
    with torch.no_grad():
        before = float(train_m4.episode_loss(net, val, DT, SCALES))
    sampler = train_m4.EpisodeSampler(arrays, task_runs, CFG_TR, DT, seed=2)
    hist = train_m4.train(net, sampler, val, CFG_TR, DT, SCALES, log=lambda *a: None)
    with torch.no_grad():
        after = float(train_m4.episode_loss(net, val, DT, SCALES))
    assert after < before and after == pytest.approx(min(h["val_loss"] for h in hist), rel=1e-5)


def test_checkpoints_hold_the_best_weights_and_the_history(tmp_path):
    import json
    import pandas as pd
    runs, arrays, task_runs, _ = data()
    torch.manual_seed(0)
    net = nets.AdaptNet(CFG_NET, CFG_M["idm"]["params"])
    val = train_m4.EpisodeSampler(arrays, task_runs, CFG_TR, DT, seed=99).sample(16)
    sampler = train_m4.EpisodeSampler(arrays, task_runs, CFG_TR, DT, seed=2)
    hist = train_m4.train(net, sampler, val, CFG_TR, DT, SCALES, log=lambda *a: None,
                          checkpoint_dir=tmp_path)
    saved = torch.load(tmp_path / "net_best.pt")
    assert all(torch.equal(saved[k], v.cpu()) for k, v in net.state_dict().items())
    assert len(pd.read_csv(tmp_path / "history.csv")) == len(hist)
    info = json.loads((tmp_path / "checkpoint.json").read_text())
    assert info["best_val_loss"] == pytest.approx(min(h["val_loss"] for h in hist))
    assert info["step"] == hist[-1]["step"]
    assert not list(tmp_path.glob("*.tmp"))


def test_a_non_finite_batch_is_skipped_and_does_not_corrupt_the_weights():
    runs, arrays, task_runs, _ = data()

    class PoisonFirst(train_m4.EpisodeSampler):
        calls = 0

        def sample(self, n):
            eps = super().sample(n)
            PoisonFirst.calls += 1
            if PoisonFirst.calls == 1:
                ctx, targets = eps[0]
                targets[0] = {**targets[0], "s": targets[0]["s"] * np.nan}
            return eps

    torch.manual_seed(0)
    net = nets.AdaptNet(CFG_NET, CFG_M["idm"]["params"])
    before = copy.deepcopy(net.state_dict())
    val = train_m4.EpisodeSampler(arrays, task_runs, CFG_TR, DT, seed=99).sample(8)
    sampler = PoisonFirst(arrays, task_runs, CFG_TR, DT, seed=2)
    one = {**CFG_TR, "steps": 1, "val_every": 1}
    hist = train_m4.train(net, sampler, val, one, DT, SCALES, log=lambda *a: None)
    assert hist[0]["skipped"] == 1
    assert all(torch.equal(before[k], v) for k, v in net.state_dict().items())
    more = {**CFG_TR, "steps": 5, "val_every": 5}
    hist = train_m4.train(net, sampler, val, more, DT, SCALES, log=lambda *a: None)
    assert hist[-1]["skipped"] == 0 and np.isfinite(hist[-1]["val_loss"])


def test_predict_params_returns_floats_within_bounds():
    runs, arrays, task_runs, _ = data()
    torch.manual_seed(0)
    net = nets.AdaptNet(CFG_NET, CFG_M["idm"]["params"])
    rid = next(iter(arrays))
    p = train_m4.predict_params(net, [arrays[rid]], DT, SCALES)
    assert set(p) == {"v0", "T", "s0", "a", "b", "delta"} and all(isinstance(v, float) for v in p.values())
    assert 0.3 <= p["T"] <= 4.0
