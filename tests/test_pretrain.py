import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ema import pretrain, train_m4
from test_m4_net import CFG_M, CFG_NET, CFG_P, CFG_TR, DT, data


def test_split_vehicles_is_disjoint_and_reproducible():
    tasks = [f"t{i}" for i in range(20)]
    tr1, va1 = pretrain.split_vehicles(tasks, 0.1, 5)
    tr2, va2 = pretrain.split_vehicles(tasks, 0.1, 5)
    assert (tr1, va1) == (tr2, va2) and len(va1) == 2 and not set(tr1) & set(va1)
    assert sorted(tr1 + va1) == sorted(tasks)


def test_targets_rmse_with_true_parameters_is_small_on_noise_free_data():
    runs, arrays, task_runs, params = data()
    items = pretrain.fixed_episodes(arrays, task_runs, list(task_runs)[:5], [10.0], CFG_TR, DT, 0)
    episodes = [e for _, _, e in items]
    p = {k: torch.tensor([float(params.loc[t, k]) for t, _, _ in items]) for k in params.columns}
    r = train_m4.targets_rmse(p, episodes, DT)
    assert r.shape == (len(items),) and float(r.max()) < 2.0      # measurement noise only


def test_run_writes_all_outputs(tmp_path):
    cfg_pre = {**CFG_P, "network": CFG_NET,
               "training": {**CFG_TR, "steps": 10, "val_every": 5, "val_share": 0.2,
                            "val_episodes": 8, "device": "cpu"}}
    summary = pretrain.run(tmp_path, cfg_pre, CFG_M, DT, log=lambda *a: None)
    for f in ["net.pt", "history.csv", "diagnostics.csv", "diagnostics_episodes.csv",
              "run_info.json"]:
        assert (tmp_path / f).exists()
    info = json.loads((tmp_path / "run_info.json").read_text())
    assert info["n_val"] >= 1 and info["steps_run"] == 10
    assert set(summary["length_s"]) == set(float(x) for x in CFG_TR["context_s"])
    assert {"rmse_pred", "rmse_true", "rmse_pop", "spearman_T"} <= set(summary.columns)
    assert (summary["rmse_true"] < summary["rmse_pop"]).all()
    assert info["device"] == "cpu" and info["resumed_from"] is None


def test_resume_starts_from_the_saved_weights(tmp_path):
    cfg_pre = {**CFG_P, "network": CFG_NET,
               "training": {**CFG_TR, "steps": 5, "val_every": 5, "val_share": 0.2,
                            "val_episodes": 8, "device": "cpu", "resume_lr": 1e-4}}
    pretrain.run(tmp_path / "a", cfg_pre, CFG_M, DT, log=lambda *a: None)
    saved = torch.load(tmp_path / "a" / "net.pt")
    zero = {**cfg_pre, "training": {**cfg_pre["training"], "steps": 0}}
    pretrain.run(tmp_path / "b", zero, CFG_M, DT, log=lambda *a: None,
                 resume_from=tmp_path / "a" / "net.pt")
    again = torch.load(tmp_path / "b" / "net.pt")
    assert all(torch.equal(saved[k], again[k]) for k in saved)
    info = json.loads((tmp_path / "b" / "run_info.json").read_text())
    assert info["resumed_from"].endswith("net.pt") and info["lr"] == 1e-4
    first = json.loads((tmp_path / "a" / "run_info.json").read_text())
    assert first["lr"] == CFG_TR["lr"] and first["n_skipped_steps"] == 0


def test_missing_gpu_gives_a_clear_error():
    if torch.cuda.is_available():
        pytest.skip("a GPU is present")
    with pytest.raises(RuntimeError, match="no GPU"):
        train_m4.resolve_device("cuda")
