import numpy as np
import pandas as pd

from ema import diagnostics, protocol
from test_population_prior import CFG_E, CFG_M, setup


def test_task_split_matches_the_validation_configurations():
    rt, folds, asg, ct, traj_fd = setup()
    for task in rt["task"].unique():
        train, val = diagnostics.task_split(rt, task, CFG_E)
        cfgs = protocol.validation_configs(rt, task, CFG_E)
        assert set(rt.loc[val, "config_id"]) <= cfgs and not set(rt.loc[train, "config_id"]) & cfgs
        assert sorted(train + val) == sorted(rt.index[rt["task"] == task])


def test_idm_floor_on_idm_generated_data():
    rt, folds, asg, ct, traj_fd = setup()
    runs_by_id = {rid: g.reset_index(drop=True) for rid, g in traj_fd.groupby("run_id")}
    task = sorted(rt["task"].unique())[0]
    rows = pd.DataFrame(diagnostics.idm_floor(task, traj_fd, rt, runs_by_id, CFG_M, CFG_E))
    train, val = diagnostics.task_split(rt, task, CFG_E)
    assert len(rows) == 2 * len(val) and set(rows["floor"]) == {"in_sample", "cross_config"}
    assert set(rows["run_id"]) == set(val) and not rows["collided"].any()
    # the data come from one IDM per vehicle, so both floors are small
    assert rows["spacing_rmse"].max() < 1.0
    assert {"idm_T", "idm_a", "n_train_runs"} <= set(rows.columns)


def test_idm_floor_is_empty_without_validation_runs():
    rt, folds, asg, ct, traj_fd = setup()
    task = sorted(rt["task"].unique())[0]
    one = rt[rt["task"] == task].iloc[:1]           # a task with a single configuration
    rows = diagnostics.idm_floor(task, traj_fd, one, {}, CFG_M, CFG_E)
    assert rows == []
