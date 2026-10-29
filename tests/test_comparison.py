import numpy as np
import pandas as pd
import pytest

from ema import config
from ema import comparison as st

LENGTHS = [10.0, 30.0]


def runs_table(offset=0.0, with_context=True, seed=0):
    """Synthetic run results: 9 vehicles (3 EV), 2 tasks each, 2 or 3 draws, 1-3 runs."""
    rng = np.random.default_rng(seed)
    rows = []
    for v in range(9):
        pt = "EV" if v < 3 else "ICE"
        for g in ["g1", "g2"]:
            task = f"{pt} | V{v} | {g}"
            for draw in range(2 + (v % 2)):
                for L in (LENGTHS if with_context else [np.nan]):
                    for r in range(1 + (draw % 3)):
                        rows.append(dict(level="L2_new_vehicle", task=task, draw=draw, length_s=L,
                                         run_id=f"{task}_{r}", collided=False,
                                         spacing_rmse=v + offset + rng.normal(0, 0.1)))
    return pd.DataFrame(rows)


def test_the_config_file_has_the_agreed_design():
    c = config.load_yaml("statistics.yaml")
    assert (c["n_boot"], c["ci"], c["min_vehicles_for_intervals"]) == (10000, 0.95, 9)
    assert c["confirmatory"]["level"] == "L2_new_vehicle"
    assert c["confirmatory"]["pairs"] == [["M2", "M0"], ["M2", "M1"], ["M4A_ft", "M0"], ["M4A_ft", "M2"]]
    assert len(c["methods"]) == 12 and c["lengths"] == [10, 30, 60, 120, 300]


def test_vehicle_values_average_runs_then_draws_then_tasks():
    runs = pd.DataFrame([
        dict(level="L", task="EV | A | g1", draw=0, length_s=10.0, run_id="r1", spacing_rmse=1.0),
        dict(level="L", task="EV | A | g1", draw=0, length_s=10.0, run_id="r2", spacing_rmse=3.0),
        dict(level="L", task="EV | A | g1", draw=1, length_s=10.0, run_id="r1", spacing_rmse=6.0),
        dict(level="L", task="EV | A | g2", draw=0, length_s=10.0, run_id="r3", spacing_rmse=np.nan),
        dict(level="L", task="EV | A | g2", draw=0, length_s=10.0, run_id="r4", spacing_rmse=10.0)])
    v = st.vehicle_values(runs)
    # draws of g1: 2 and 6 -> task 4; g2: 10 (collided run ignored) -> vehicle (4 + 10) / 2 = 7
    assert len(v) == 1 and v["value"].iloc[0] == pytest.approx(7.0)
    assert (v["vehicle"].iloc[0], v["powertrain"].iloc[0]) == ("A", "EV")


def test_m0_is_repeated_at_every_length_and_pairs_are_by_vehicle():
    a = st.vehicle_values(runs_table(offset=-0.5))
    m0 = st.vehicle_values(runs_table(with_context=False, seed=1))
    d = st.paired_differences(a, m0, "L2_new_vehicle", LENGTHS)
    assert len(d) == 9 * len(LENGTHS) and set(d["length_s"]) == set(LENGTHS)
    assert d["d"].mean() == pytest.approx(-0.5, abs=0.1)


def test_bootstrap_is_reproducible_and_covers_the_mean():
    d = np.array([0.3, -0.1, 0.5, 0.2, 0.4, 0.0, 0.6, 0.1, 0.25])
    assert st.bootstrap(d, 2000, 0.95, 7) == st.bootstrap(d, 2000, 0.95, 7)
    mean, lo, hi, p = st.bootstrap(d, 2000, 0.95, 7)
    assert lo < mean < hi and mean == pytest.approx(d.mean()) and p < 0.01
    _, _, _, p_all = st.bootstrap(np.ones(9), 999, 0.95, 1)
    assert p_all == pytest.approx(2 / 1000)                   # never exactly 0
    rng, hits = np.random.default_rng(3), 0
    for i in range(200):                                      # coverage of a true mean of 1.0
        _, lo, hi, _ = st.bootstrap(rng.normal(1.0, 1.0, 9), 1000, 0.95, i)
        hits += lo <= 1.0 <= hi
    assert 0.80 <= hits / 200 <= 0.99                         # percentile bootstrap, n = 9


def test_holm_matches_a_hand_calculation():
    p = [0.01, 0.04, 0.03, 0.005]
    # sorted 0.005*4 = 0.02, 0.01*3 = 0.03, 0.03*2 = 0.06, 0.04*1 = 0.04 -> monotone 0.06
    assert np.allclose(st.holm(p), [0.03, 0.06, 0.06, 0.02])
    assert np.allclose(st.holm([0.5, 0.9]), [1.0, 1.0])


def test_compare_gives_p_values_only_for_confirmatory_pairs():
    values = {"A": st.vehicle_values(runs_table(offset=-0.5)),
              "B": st.vehicle_values(runs_table(seed=1)),
              "M0": st.vehicle_values(runs_table(with_context=False, seed=2))}
    cfg = {"n_boot": 500, "ci": 0.95, "seed": 3, "min_vehicles_for_intervals": 9}
    conf = st.compare(values, [("A", "B"), ("A", "M0")], "L2_new_vehicle", LENGTHS, cfg, "confirmatory")
    assert len(conf) == 4 and conf["p"].notna().all() and (conf["p_holm"] >= conf["p"]).all()
    assert (conf["n_better"] == 9).all() and (conf["ci_hi"] < 0).all()
    expl = st.compare(values, [("A", "B")], "L2_new_vehicle", LENGTHS, cfg, "exploratory")
    assert expl["p"].isna().all() and expl["p_holm"].isna().all()
    again = st.compare(values, [("A", "B")], "L2_new_vehicle", LENGTHS, cfg, "exploratory")
    pd.testing.assert_frame_equal(expl, again)
    few = st.compare(values, [("A", "B")], "L2_new_vehicle", LENGTHS, {**cfg, "min_vehicles_for_intervals": 10},
                     "exploratory")
    assert len(few) == 0                                      # fewer than 10 vehicles: no interval
    assert len(st.compare(values, [("A", "missing")], "L2_new_vehicle", LENGTHS, cfg, "exploratory")) == 0


def test_collisions_are_counted_by_run_type():
    r = runs_table().assign(collided=False)
    r.loc[r.index[:3], "collided"] = True
    is_stop = pd.Series({rid: rid.endswith("_0") for rid in r["run_id"].unique()})
    c = st.collisions({"M": r}, is_stop)
    assert c["n_collided"].sum() == 3 and c["n_runs"].sum() == len(r)
    assert set(c["run_type"]) == {"stop", "no_stop"}
