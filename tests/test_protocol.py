import os

import pandas as pd
import pytest

from ema import config, data, protocol

CFG = {
    "runs": {"qc_pass_only": True, "include_stop_runs": True},
    "task_keys": ["powertrain", "vehicle", "gap"],
    "config_keys": ["vehicle", "gap", "profile", "desired_mph", "v_free_mph", "v_reduced_mph"],
    "levels": ["L1_new_setting", "L2_new_vehicle", "L3_cross_powertrain", "L2_no_ice"],
    "validation": {"fraction_of_configs": 0.34, "seed": 42},
}


def synthetic_runs():
    """EV vehicle A (2 gaps), EV B (1 gap), ICE C and D (2 gaps); 3 configs per task,
    the middle one repeated twice; plus one run that fails QC."""
    rows = []
    spec = [("EV", "A", ["S", "L"]), ("EV", "B", ["S"]),
            ("ICE", "C", ["min", "max"]), ("ICE", "D", ["min", "max"])]
    for pt, veh, gaps in spec:
        for gap in gaps:
            for k, v_red in enumerate([0, 25, 45]):
                for rep in ([1, 2] if k == 1 else [1]):
                    rows.append(dict(run_id=f"{pt}_{veh}_{gap}_{v_red}_r{rep}", powertrain=pt,
                                     vehicle=veh, gap=gap, profile="dip", desired_mph=0.0,
                                     v_free_mph=55.0, v_reduced_mph=float(v_red),
                                     qc_pass=True, exclude=False))
    rows.append(dict(run_id="EV_A_S_bad", powertrain="EV", vehicle="A", gap="S", profile="dip",
                     desired_mph=0.0, v_free_mph=55.0, v_reduced_mph=35.0,
                     qc_pass=False, exclude=False))
    return pd.DataFrame(rows)


def setup():
    rt = protocol.run_table(synthetic_runs(), CFG)
    return rt, protocol.make_folds(rt, CFG)


def test_run_table_filters_qc_and_flags_stop_runs():
    rt, _ = setup()
    assert "EV_A_S_bad" not in rt.index
    assert rt.loc["EV_A_S_0_r1", "is_stop_run"]
    assert not rt.loc["ICE_C_min_0_r1", "is_stop_run"]


def test_run_table_can_drop_stop_runs():
    cfg = {**CFG, "runs": {"qc_pass_only": True, "include_stop_runs": False}}
    assert not protocol.run_table(synthetic_runs(), cfg)["is_stop_run"].any()


def test_fold_counts_per_level():
    _, folds = setup()
    counts = pd.Series([f["level"] for f in folds]).value_counts().to_dict()
    assert counts == {"L1_new_setting": 6, "L2_new_vehicle": 4,
                      "L3_cross_powertrain": 1, "L2_no_ice": 2}


def test_folds_are_disjoint_and_respect_levels():
    rt, folds = setup()
    tasks = rt.groupby("task")[["vehicle", "powertrain"]].first()
    for f in folds:
        assert not set(f["train"]) & set(f["test"])
        train_info, test_info = tasks.loc[f["train"]], tasks.loc[f["test"]]
        if f["level"] == "L2_new_vehicle":
            assert not set(test_info["vehicle"]) & set(train_info["vehicle"])
        if f["level"] == "L3_cross_powertrain":
            assert set(train_info["powertrain"]) == {"ICE"}
            assert set(test_info["powertrain"]) == {"EV"}
        if f["level"] == "L2_no_ice":
            assert set(train_info["powertrain"]) == {"EV"}


def test_configurations_are_never_split():
    rt, folds = setup()
    a = protocol.assign_runs(rt, folds, CFG)
    assert (a.groupby(["fold_id", "config_id"])["role"].nunique() == 1).all()


def test_validation_is_identical_across_folds_and_nonempty():
    rt, folds = setup()
    a = protocol.assign_runs(rt, folds, CFG)
    fit = a[a["role"].isin(["train", "val"])]
    per = fit.groupby(["task", "fold_id"]).apply(
        lambda g: frozenset(g.loc[g["role"] == "val", "run_id"]), include_groups=False)
    assert (per.groupby(level="task").nunique() == 1).all()
    assert (per.map(len) >= 1).all()


def test_assignment_is_deterministic():
    rt, folds = setup()
    pd.testing.assert_frame_equal(protocol.assign_runs(rt, folds, CFG),
                                  protocol.assign_runs(rt, folds, CFG))


needs_data = pytest.mark.skipif(not os.environ.get(config.ENV_DATA_ROOT),
                                reason="EMA_DATA_ROOT not set")


@needs_data
def test_real_data_tasks_and_fold_counts():
    cfg = config.load_yaml("evaluation.yaml")
    rt = protocol.run_table(data.load_runs(), cfg)
    assert len(rt) == 332 and rt["task"].nunique() == 18
    counts = pd.Series([f["level"] for f in protocol.make_folds(rt, cfg)]).value_counts().to_dict()
    assert counts == {"L1_new_setting": 16, "L2_new_vehicle": 9,
                      "L3_cross_powertrain": 1, "L2_no_ice": 3}
