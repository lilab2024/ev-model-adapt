import hashlib
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from ema import config, evaluate, methods, nets, pretrain
from test_m4_lovo import FT, four_vehicle_setup
from test_m4_net import CFG_M, CFG_NET, CFG_P, CFG_TR, DT

CFG_TF = {**CFG_NET, "encoder": "transformer", "d_model": 16, "layers": 1, "heads": 2, "ff": 32}
SPEC = CFG_M["idm"]["params"]


def n_weights(net):
    return sum(p.numel() for p in net.parameters())


def test_the_config_file_matches_the_gru_network_except_the_encoder():
    tf, gru = config.load_yaml("transformer.yaml")["network"], config.load_yaml("pretraining.yaml")["network"]
    assert tf["encoder"] == "transformer" and tf["scales"] == gru["scales"]
    assert (tf["z_dim"], tf["head_hidden"]) == (gru["z_dim"], gru["head_hidden"])
    assert (tf["d_model"], tf["layers"], tf["heads"], tf["ff"]) == (32, 2, 4, 64)
    big_tf = nets.AdaptNet(tf, config.load_yaml("models.yaml")["idm"]["params"])
    big_gru = nets.AdaptNet(gru, config.load_yaml("models.yaml")["idm"]["params"])
    assert n_weights(big_gru) == 14861 and 15000 < n_weights(big_tf) < 22000


def test_gru_network_is_unchanged():
    net = nets.AdaptNet(CFG_NET, SPEC)
    assert {k.split(".")[0] for k in net.state_dict()} == {"log_lo", "log_hi", "gru", "to_z", "head"}


def test_transformer_ignores_padding_and_gives_valid_parameters():
    torch.manual_seed(0)
    net = nets.AdaptNet(CFG_TF, SPEC).eval()
    rng = np.random.default_rng(0)
    pieces = [rng.normal(size=(n, 4)).astype(np.float32) for n in (30, 50, 20)]
    x, lengths, episode = nets.collate_contexts([pieces[:2], pieces[2:]])
    with torch.no_grad():
        p1 = net(x, lengths, episode, 2)
        junk = x.clone()
        for j, n in enumerate(lengths.tolist()):
            junk[j, n:] = 99.0                                 # padding content must not matter
        p2 = net(junk, lengths, episode, 2)
        alone = net(*nets.collate_contexts([pieces[2:]]), 1)   # other episodes must not matter
    for k in p1:
        assert torch.allclose(p1[k], p2[k], atol=1e-6) and torch.allclose(p1[k][1:], alone[k], atol=1e-6)
    lo, hi = torch.exp(net.log_lo), torch.exp(net.log_hi)
    for i, k in enumerate(net.free):
        assert torch.all(p1[k] >= lo[i] - 1e-6) and torch.all(p1[k] <= hi[i] + 1e-6)
    torch.manual_seed(0)
    again = nets.AdaptNet(CFG_TF, SPEC).eval()
    with torch.no_grad():
        p3 = again(x, lengths, episode, 2)
    assert all(torch.equal(p1[k], p3[k]) for k in p1)


def test_pretraining_runs_with_the_transformer(tmp_path):
    cfg_pre = {**CFG_P, "network": CFG_TF,
               "training": {**CFG_TR, "steps": 6, "val_every": 3, "val_share": 0.2,
                            "val_episodes": 8, "device": "cpu"}}
    pretrain.run(tmp_path, cfg_pre, CFG_M, DT, log=lambda *a: None)
    info = json.loads((tmp_path / "run_info.json").read_text())
    assert info["network"]["encoder"] == "transformer" and info["steps_run"] == 6
    state = torch.load(tmp_path / "net.pt")
    assert any(k.startswith("tf.") for k in state)


def test_transformer_lovo_runs_from_its_pretrained_weights(tmp_path):
    from test_context_recalibration import CFG_M as CFG_M2
    torch.manual_seed(1)
    net = nets.AdaptNet(CFG_TF, CFG_M2["idm"]["params"])
    path = tmp_path / "tf.pt"
    torch.save(net.state_dict(), path)
    cfg_tf = {"network": CFG_TF, "pretrained": {"weights": str(path),
                                                "md5": hashlib.md5(path.read_bytes()).hexdigest()}}
    cfg_m4 = {"context_piece_s": 6.0, "eval_device": "cpu", "finetuning": FT}
    cfg_e, rt, folds, asg, ct, traj_fd = four_vehicle_setup(["L2_new_vehicle"])
    make = lambda cfg_m, c: methods.M4TransformerLOVO(cfg_m, cfg_e, cfg_m4, {"network": CFG_NET},
                                                      log=lambda *a: None, cfg_tf=cfg_tf)
    res, st = evaluate.run_evaluation(make, CFG_M2, rt, folds, asg, ct, traj_fd, cfg_e, verbose=False)
    assert all(s["method"] == "m4a_transformer_lovo" and s["lovo_used"] for s in st)
    assert all(s["weights_md5"] == cfg_tf["pretrained"]["md5"] for s in st)
    res2, _ = evaluate.run_evaluation(make, CFG_M2, rt, folds, asg, ct, traj_fd, cfg_e, verbose=False)
    pd.testing.assert_frame_equal(res, res2)
    bad = {**cfg_tf, "pretrained": {**cfg_tf["pretrained"], "md5": "none"}}
    with pytest.raises(ValueError, match="md5"):
        methods.M4TransformerLOVO(CFG_M2, cfg_e, cfg_m4, {"network": CFG_NET}, cfg_tf=bad)
