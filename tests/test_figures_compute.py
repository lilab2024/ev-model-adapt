"""Computations of the group-2 figures (D050): the parts that need no data. The full
recomputations are checked against the recorded runs whenever the figures are made."""
import numpy as np
import pandas as pd
import pytest

from ema import config, figures_compute as FC
from ema.models import idm

CFG_M = config.load_yaml("models.yaml")


def test_first_collision_fixed_order():
    runs = pd.DataFrame(dict(level=["L3_cross_powertrain"] * 4 + ["L2_no_ice"],
                             task=["EV | B | Short", "EV | A | Short", "EV | A | Short", "EV | A | Short", "EV | A | Short"],
                             draw=[0, 1, 0, 0, 0], length_s=[10.0, 10.0, 30.0, 10.0, 10.0],
                             run_id=["r1", "r2", "r3", "r4", "r0"], collided=[True, True, True, False, True]))
    row, n = FC.first_collision(runs)
    assert row["run_id"] == "r3" and n == 3                  # task, draw, length, run; not collided / other level out
    with pytest.raises(FC.ReproductionError):
        FC.first_collision(runs.assign(collided=False))


def test_density_level_holds_the_mass():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=20000), rng.normal(size=20000)
    edges = np.linspace(-5, 5, 61)
    H, lev = FC.density_level(x, y, edges, edges, 0.9)
    assert H.shape == (60, 60)
    share = H[H >= lev].sum() / H.sum()
    assert 0.9 <= share < 0.93


def test_smoothed_density_and_rel_diff():
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=3000), rng.normal(size=3000)
    edges = np.linspace(-5, 5, 31)
    H, lev = FC.density_level(x, y, edges, edges, 0.9, smooth=1.2)
    assert 0.9 <= H[H >= lev].sum() / H.sum() < 0.95
    assert FC._rel_diff(1.0, 1.0) == 0.0 and FC._rel_diff(np.nan, np.nan) == 0.0
    assert FC._rel_diff(1.0, np.nan) == np.inf and 0 < FC._rel_diff(1.0, 1.0 + 1e-15) < 2e-15


def test_idm_accel_grid_matches_the_controller():
    p = dict(v0=33.0, T=1.2, s0=2.0, a=1.0, b=1.5, delta=4.0)
    gaps, dvs = np.array([5.0, 20.0, 50.0]), np.array([-2.0, 0.0, 1.0])
    A = FC.idm_accel_grid(p, 15.0, gaps, dvs, 0.1)
    c = idm.IDMController(p, 0.1)
    assert A.shape == (3, 3) and A[2, 1] == c.accel(20.0, 15.0, 16.0)


def test_floor_targets_one_fit_per_task():
    d = pd.DataFrame(dict(task=["t1", "t1", "t2"], floor="cross_config", idm_v0=[30.0, 30.0, 31.0],
                          idm_T=1.0, idm_s0=2.0, idm_a=1.0, idm_b=1.5))
    t = FC.floor_targets(d, ["v0", "T", "s0", "a", "b"])
    assert t["t2"]["v0"] == 31.0 and set(t) == {"t1", "t2"}
    with pytest.raises(ValueError):
        FC.floor_targets(d.assign(idm_v0=[30.0, 29.0, 31.0]), ["v0", "T", "s0", "a", "b"])


def test_make_context_needs_rows():
    ct = pd.DataFrame(dict(task=["t"], draw=[0], role=["context"], length_s=[10.0], run_id=["r"], t0=[0.0],
                           t1=[10.0]))
    runs = {"r": pd.DataFrame(dict(t=np.arange(0, 20, 0.1), s=1.0, v_f=1.0, v_l=1.0))}
    c = FC.make_context(ct, runs, "t", 0, 10.0)
    assert c["length_s"] == 10.0 and len(c["segments"]) == 1 and len(c["segments"][0]) == 101
    with pytest.raises(FC.ReproductionError):
        FC.make_context(ct, runs, "t", 1, 10.0)


def test_residual_grid_untrained_is_zero():
    torch = pytest.importorskip("torch")
    from ema import residual
    cfg_res = config.load_yaml("residual.yaml")
    torch.manual_seed(0)
    net = residual.ResidualNet(cfg_res["network"]["hidden"], cfg_res["network"]["a_max"])
    w = residual.numpy_weights(net)                         # last layer starts at zero
    p = dict(v0=33.0, T=1.2, s0=2.0, a=1.0, b=1.5, delta=4.0)
    R = FC.residual_grid(w, 0.5, residual.scale_vector(cfg_res), p, 15.0, np.linspace(5, 50, 7),
                         np.linspace(-3, 3, 5), 0.1)
    assert R.shape == (5, 7) and np.all(R == 0.0)
    with torch.no_grad():
        net.out.bias.fill_(0.3)
    w = residual.numpy_weights(net)
    R = FC.residual_grid(w, 0.5, residual.scale_vector(cfg_res), p, 15.0, np.linspace(5, 50, 7),
                         np.linspace(-3, 3, 5), 0.1)
    assert np.allclose(R, 0.5 * np.tanh(0.3))


def test_inject_targets_clears_the_task_fit_cache():
    """D053: a fold uses only the injected fits, whatever earlier calls left in the cache."""
    from ema import methods
    cfg_e = config.load_yaml("evaluation.yaml")
    rt = pd.DataFrame(dict(task=["EV | A | Short", "EV | A | Short", "ICE | B | min"]),
                      index=pd.Index(["r1", "r2", "r3"], name="run_id"))
    a = pd.DataFrame(dict(fold_id=["f1"] * 3, role=["train"] * 3, run_id=["r1", "r2", "r3"]))
    prot = dict(assignment=a, rt=rt)
    targets = {"EV | A | Short": {"T": 1.1}, "ICE | B | min": {"T": 2.2}}
    cache = methods.PopulationPrior._task_fits
    cfg_key = methods.PopulationPrior(CFG_M, cfg_e)._cfg_key()
    stale = ("EV | A | Short", ("r1", "r2"), cfg_key)
    try:
        cache.clear()
        cache[stale] = {"T": 9.9}                                   # left by an earlier fall-back
        cache[("other", ("x",), cfg_key)] = {"T": 5.0}
        keys = FC._inject_targets(prot, dict(fold_id="f1"), targets, CFG_M, cfg_e)
        assert sorted(keys) == sorted(cache) and len(cache) == 2
        assert cache[stale] == {"T": 1.1} and cache[("ICE | B | min", ("r3",), cfg_key)] == {"T": 2.2}
    finally:
        cache.clear()


def test_read_exact_keeps_the_last_bit(tmp_path):
    """D054: the default pandas parser reads this value one bit off; read_exact does not."""
    p = tmp_path / "x.csv"
    p.write_text("b" + chr(10) + "3.1719959010078993" + chr(10))
    assert FC.read_exact(p)["b"].iloc[0] == float("3.1719959010078993")
    x = float("3.1719959010078993")
    pd.DataFrame(dict(b=[x, 0.1 + 0.2, 1 / 3])).to_csv(p, index=False)
    assert list(FC.read_exact(p)["b"]) == [x, 0.1 + 0.2, 1 / 3]          # what to_csv writes comes back
