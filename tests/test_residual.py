import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from ema import evaluate, methods, residual, simulate, torch_sim
from ema.models import idm
from test_population_prior import CFG_M, DT
from test_population_prior_same_pt import setup

CFG_R = {"network": {"hidden": [8, 8], "a_max": 0.5},
         "scales": {"gap": 30.0, "v": 30.0, "dv": 5.0, "a_idm": 2.0, "a_lead": 2.0, "a_prev": 2.0},
         "training": {"window_s": 10.0, "batch_windows": 4, "penalty": 1.0, "lr": 1e-2,
                      "grad_clip": 1.0, "val_every": 3, "patience": 5, "max_steps": 6,
                      "val_windows_per_task": 2, "device": "cpu", "seed": 3}}
P = dict(v0=33.0, T=1.5, s0=2.5, a=1.0, b=1.5, delta=4.0)
SCALES = residual.scale_vector(CFG_R)


def windows(n=3, length=120):
    rng = np.random.default_rng(0)
    t = np.arange(length) * DT
    out = []
    for i in range(n):
        v_lead = 20.0 - 4.0 * np.exp(-((t - 5.0 - i) / 2.0) ** 2)
        sim = simulate.rollout(idm.IDMController(P), v_lead, s0=35.0, v0=20.0, dt=DT)
        out.append({"s": sim["s"] + rng.normal(0, 0.2, length), "v_f": sim["v"], "v_l": v_lead})
    return out


def random_net(scale=0.5, seed=1):
    torch.manual_seed(seed)
    net = residual.ResidualNet([8, 8], 0.5).double()
    with torch.no_grad():
        for prm in net.parameters():
            prm.copy_(torch.randn_like(prm) * scale)
    return net


def test_the_config_file_has_the_agreed_settings():
    from ema import config
    c = config.load_yaml("residual.yaml")
    assert c["network"] == {"hidden": [64, 64], "a_max": 0.5}
    assert c["scales"] == CFG_R["scales"]
    t = c["training"]
    assert (t["window_s"], t["batch_windows"], t["penalty"], t["lr"], t["grad_clip"]) == (60.0, 64, 1.0, 1e-3, 1.0)
    assert (t["val_every"], t["patience"], t["max_steps"], t["val_windows_per_task"]) == (50, 10, 3000, 16)


def test_untrained_network_is_zero_and_output_is_bounded():
    net = residual.ResidualNet([8, 8], 0.5)
    x = torch.randn(50, 6)
    assert torch.all(net(x) == 0)
    big = random_net(scale=20.0)
    assert torch.all(big(torch.randn(500, 6, dtype=torch.float64) * 10).abs() <= 0.5)


def test_numpy_forward_equals_torch():
    net = random_net()
    x = np.random.default_rng(1).normal(size=(20, 6))
    a = net(torch.as_tensor(x)).detach().numpy()
    b = residual.numpy_forward(residual.numpy_weights(net), 0.5, x)
    assert np.allclose(a, b, atol=1e-12)


def test_zero_correction_rollout_equals_the_idm_rollout():
    ws = windows()
    seg = residual.pack_windows(ws, DT, dtype=torch.float64)
    p = residual.params_tensor([P] * len(ws), dtype=torch.float64)
    rmse, pen, s = residual.rollout(residual.ResidualNet([8, 8], 0.5).double(), p, seg, DT, SCALES)
    ref = torch_sim.rollout(p, seg, DT)
    assert torch.allclose(s, ref["s"]) and torch.all(pen == 0)
    assert torch.allclose(rmse, torch_sim.spacing_rmse(ref["s"], seg), atol=1e-6)


def test_training_rollout_equals_the_evaluation_controller():
    net = random_net(scale=0.3)
    ws = windows()
    seg = residual.pack_windows(ws, DT, dtype=torch.float64)
    p = residual.params_tensor([P] * len(ws), dtype=torch.float64)
    _, _, s = residual.rollout(net, p, seg, DT, SCALES)
    for i, w in enumerate(ws):
        ctrl = residual.ResidualIDMController(P, residual.numpy_weights(net), 0.5, SCALES, DT)
        lp = simulate.leader_position(w["s"], w["v_f"], DT)
        sim = simulate.rollout(ctrl, w["v_l"], w["s"][0], w["v_f"][0], DT, lead_pos=lp)
        assert np.allclose(sim["s"], s[i].detach().numpy(), atol=1e-8)


def make(**train):
    cfg = {**CFG_R, "training": {**CFG_R["training"], **train}}
    return lambda cfg_m, cfg_e: methods.M2bResidual(cfg_m, cfg_e, cfg, log=lambda *a: None)


def test_m2b_residual_runs_and_equals_m2b_without_learning():
    methods.PopulationPrior._task_fits.clear()
    cfg_e, rt, folds, asg, ct, traj_fd = setup(["L2_new_vehicle"])
    res, st = evaluate.run_evaluation(make(), CFG_M, rt, folds, asg, ct, traj_fd, cfg_e,
                                      verbose=False)
    for f, s in zip(folds, st):
        pt = f["test"][0].split(" | ")[0]
        assert s["method"] == "m2b_residual" and s["prior_powertrain"] == pt
        assert s["res_heldout_vehicle"] != f["test"][0].split(" | ")[1]
        assert s["res_stage2_steps"] == s["res_best_step"] <= CFG_R["training"]["max_steps"]
    res2, _ = evaluate.run_evaluation(make(), CFG_M, rt, folds, asg, ct, traj_fd, cfg_e,
                                      verbose=False)
    pd.testing.assert_frame_equal(res, res2)                  # reproducible
    still, st0 = evaluate.run_evaluation(make(lr=0.0), CFG_M, rt, folds, asg, ct, traj_fd, cfg_e,
                                         verbose=False)
    m2b, _ = evaluate.run_evaluation(methods.PopulationPriorSamePowertrain, CFG_M, rt, folds, asg,
                                     ct, traj_fd, cfg_e, verbose=False)
    assert all(s["res_best_step"] == 0 for s in st0)
    keys = ["task", "draw", "length_s", "run_id"]
    a, b = still.sort_values(keys).reset_index(drop=True), m2b.sort_values(keys).reset_index(drop=True)
    assert np.array_equal(a["spacing_rmse"].to_numpy(), b["spacing_rmse"].to_numpy())


def test_m2b_residual_saves_its_artifacts(tmp_path):
    methods.PopulationPrior._task_fits.clear()
    cfg_e, rt, folds, asg, ct, traj_fd = setup(["L2_new_vehicle"])
    evaluate.run_evaluation(make(), CFG_M, rt, folds[:1], asg, ct, traj_fd, cfg_e, verbose=False,
                            artifact_dir=tmp_path)
    fid = folds[0]["fold_id"]
    assert (tmp_path / "nets" / f"{fid}.pt").exists()
    h = pd.read_csv(tmp_path / "histories" / f"{fid}.csv")
    assert h["step"].iloc[0] == 0 and (tmp_path / "histories" / f"{fid}_stage2.csv").exists()


def test_m2b_residual_holds_out_a_vehicle_when_there_are_two():
    methods.PopulationPrior._task_fits.clear()
    cfg_e, rt, folds, asg, ct, traj_fd = setup(["L1_new_setting"])
    f = folds[0]                                   # EV A, one gap: training EVs are A (other gap) and B
    _, st = evaluate.run_evaluation(make(), CFG_M, rt, [f], asg, ct, traj_fd, cfg_e, verbose=False)
    s = st[0]
    assert s["res_heldout_vehicle"] in ("A", "B")
    n_same = sum(1 for t in f["train"] if t.startswith("EV"))
    held_tasks = sum(1 for t in f["train"] if t.startswith("EV") and t.split(" | ")[1] == s["res_heldout_vehicle"])
    assert s["res_stage1_tasks"] == n_same - held_tasks > 0
