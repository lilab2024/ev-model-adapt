import hashlib
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from ema import config, direct, evaluate, methods, nets, pretrain, residual, simulate, train_m4
from ema.models import idm
from test_m4_lovo import FT, four_vehicle_setup
from test_m4_net import CFG_M, CFG_NET, CFG_P, CFG_TR, DT

CFG_POL = {"hidden": [8, 8], "accel_limits": [-8.0, 4.0],
           "scales": {"gap": 30.0, "v": 30.0, "dv": 5.0, "a_lead": 2.0, "a_prev": 2.0}}
SPEC = CFG_M["idm"]["params"]
P = dict(v0=33.0, T=1.5, s0=2.5, a=1.0, b=1.5, delta=4.0)


def random_policy(scale=0.4, seed=2):
    torch.manual_seed(seed)
    net = direct.DirectNet(CFG_NET, SPEC, CFG_POL).double()
    with torch.no_grad():
        for prm in net.policy.parameters():
            prm.copy_(torch.randn_like(prm) * scale)
    return net


def windows(n=3, length=120):
    t = np.arange(length) * DT
    out = []
    for i in range(n):
        v_lead = 20.0 - 4.0 * np.exp(-((t - 5.0 - i) / 2.0) ** 2)
        sim = simulate.rollout(idm.IDMController(P), v_lead, s0=35.0, v0=20.0, dt=DT)
        out.append({"s": sim["s"], "v_f": sim["v"], "v_l": v_lead})
    return out


def test_the_config_file_has_the_agreed_settings():
    c = config.load_yaml("direct.yaml")
    assert c["policy"]["hidden"] == [64, 64] and c["policy"]["accel_limits"] == [-8.0, 4.0]
    assert c["policy"]["scales"] == CFG_POL["scales"]
    assert c["init_encoder"] == config.load_yaml("m4.yaml")["pretrained"]


def test_policy_output_stays_within_the_limits():
    net = random_policy(scale=30.0)
    a = net.policy(torch.randn(500, 5, dtype=torch.float64) * 10, torch.randn(500, CFG_NET["z_dim"], dtype=torch.float64) * 10)
    assert torch.all(a >= -8.0) and torch.all(a <= 4.0)


def test_training_rollout_equals_the_evaluation_controller():
    net = random_policy()
    ws = windows()
    seg = residual.pack_windows(ws, DT, dtype=torch.float64)
    z = torch.randn(len(ws), CFG_NET["z_dim"], dtype=torch.float64)
    _, s = direct.rollout(net.policy, z, seg, DT, net.feat_scales.double())
    for i, w in enumerate(ws):
        ctrl = direct.DirectController(direct.numpy_weights(net.policy), z[i].numpy(),
                                       CFG_POL["accel_limits"], direct.scale_vector(CFG_POL), DT)
        lp = simulate.leader_position(w["s"], w["v_f"], DT)
        sim = simulate.rollout(ctrl, w["v_l"], w["s"][0], w["v_f"][0], DT, lead_pos=lp)
        assert np.allclose(sim["s"], s[i].detach().numpy(), atol=1e-8)


def save_encoder(tmp_path):
    torch.manual_seed(5)
    enc = nets.AdaptNet(CFG_NET, SPEC)
    path = tmp_path / "enc.pt"
    torch.save(enc.state_dict(), path)
    return enc, {"policy": CFG_POL, "init_encoder": {
        "weights": str(path), "md5": hashlib.md5(path.read_bytes()).hexdigest()}}


def test_pretraining_starts_from_the_encoder_and_uses_its_own_loss(tmp_path):
    enc, cfg_d = save_encoder(tmp_path)
    net = direct.make_pretraining_net(CFG_NET, SPEC, cfg_d)
    a, b = net.enc.state_dict(), enc.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)
    with pytest.raises(ValueError, match="md5"):
        direct.make_pretraining_net(CFG_NET, SPEC, {**cfg_d, "init_encoder": {
            **cfg_d["init_encoder"], "md5": "x"}})
    cfg_pre = {**CFG_P, "network": CFG_NET,
               "training": {**CFG_TR, "steps": 6, "val_every": 3, "val_share": 0.2,
                            "val_episodes": 8, "device": "cpu"}}
    out = tmp_path / "run"
    summary = pretrain.run(out, cfg_pre, CFG_M, DT, log=lambda *a: None,
                           make_net=lambda: direct.make_pretraining_net(CFG_NET, SPEC, cfg_d),
                           extra_info={"model": "m4c_direct"})
    info = json.loads((out / "run_info.json").read_text())
    assert info["model"] == "m4c_direct" and info["steps_run"] == 6
    assert "rmse_pred" in summary.columns and not any(c.startswith("spearman") for c in summary.columns)
    assert any(k.startswith("policy.") for k in torch.load(out / "net.pt"))


def test_m4c_direct_lovo_runs_and_is_reproducible(tmp_path):
    from test_context_recalibration import CFG_M as CFG_M2
    torch.manual_seed(1)
    net = direct.DirectNet(CFG_NET, CFG_M2["idm"]["params"], CFG_POL)
    path = tmp_path / "direct.pt"
    torch.save(net.state_dict(), path)
    cfg_d = {"policy": CFG_POL, "pretrained": {"weights": str(path),
                                               "md5": hashlib.md5(path.read_bytes()).hexdigest()}}
    cfg_m4 = {"context_piece_s": 6.0, "eval_device": "cpu", "finetuning": FT}
    cfg_e, rt, folds, asg, ct, traj_fd = four_vehicle_setup(["L2_new_vehicle"])
    make = lambda cfg_m, c: methods.M4DirectLOVO(cfg_m, cfg_e, cfg_m4, {"network": CFG_NET},
                                                 log=lambda *a: None, cfg_direct=cfg_d)
    res, st = evaluate.run_evaluation(make, CFG_M2, rt, folds, asg, ct, traj_fd, cfg_e, verbose=False)
    assert all(s["method"] == "m4c_direct_lovo" and s["lovo_used"] for s in st)
    assert "ctx_n_pieces" in res.columns and len(res) > 0
    res2, _ = evaluate.run_evaluation(make, CFG_M2, rt, folds, asg, ct, traj_fd, cfg_e, verbose=False)
    pd.testing.assert_frame_equal(res, res2)
    bad = {**cfg_d, "pretrained": {**cfg_d["pretrained"], "md5": "none"}}
    with pytest.raises(ValueError, match="md5"):
        methods.M4DirectLOVO(CFG_M2, cfg_e, cfg_m4, {"network": CFG_NET}, cfg_direct=bad)
