import numpy as np
import pandas as pd
import pytest

from ema import calibrate, sim_calibrate, synthetic
from ema.models import idm

DT = 0.1
CFG_M = {"idm": {"params": {"v0": [25.0, 60.0], "T": [0.3, 4.0], "s0": [0.5, 15.0],
                            "a": [0.2, 4.0], "b": [0.01, 6.0], "delta": 4.0},
                 "min_gap_m": 0.1},
         "sim_calibration": {"segment_s": 30.0, "segments_per_task": 2, "seed": 0}}
CFG_P = {"synthetic": {
    "n_vehicles": 60, "runs_per_vehicle": 3, "run_s": 60.0,
    "leader": {"base_speed_mps": [12.0, 30.0], "kinds": ["dip", "osc", "step"],
               "dip_depth_frac": [0.2, 1.0], "dip_start_s": [5.0, 20.0], "hold_s": [0.0, 15.0],
               "dip_stop_share": 0.4,
               "osc_amplitude_mps": [1.0, 5.0], "osc_period_s": [10.0, 40.0],
               "step_size_mps": [3.0, 8.0], "step_every_s": [10.0, 25.0],
               "decel_mps2": [0.5, 2.5], "accel_mps2": [0.5, 2.0]},
    "init_gap_factor": [0.8, 1.3],
    "noise": {"spacing_m": 0.2, "speed_mps": 0.05},
    "smoothing": {"window_s": 1.0, "polyorder": 2},
    "reject": {"min_gap_m": 0.5, "max_gap_m": 150.0},
    "seed": 7}}


def test_reproducible_and_parameters_within_bounds():
    r1, t1, p1 = synthetic.make_dataset(CFG_P, CFG_M, DT)
    r2, t2, p2 = synthetic.make_dataset(CFG_P, CFG_M, DT)
    pd.testing.assert_frame_equal(t1, t2)
    for k, (lo, hi) in [(k, v) for k, v in CFG_M["idm"]["params"].items() if isinstance(v, list)]:
        assert p1[k].between(lo, hi).all()
    assert (p1["delta"] == 4.0).all()


def test_runs_are_plausible_and_well_formed():
    runs, traj, params = synthetic.make_dataset(CFG_P, CFG_M, DT, noise=False)
    assert len(runs) > 0.7 * 60 * 3                       # most runs are kept
    g = traj.groupby("run_id")
    assert g["s"].min().min() >= 0.5 and g["s"].max().max() <= 150.0
    assert (traj["v_l"] >= 0).all() and (traj["v_f"] >= 0).all()
    acc_l = g["v_l"].diff() / DT
    assert acc_l.min() >= -2.5 - 1e-9 and acc_l.max() <= 2.0 + 1e-9
    assert set(runs["kind"]) == {"dip", "osc", "step"}
    assert set(runs["task"]) == set(params.index)


def test_noise_free_records_follow_the_idm_exactly():
    runs, traj, params = synthetic.make_dataset(CFG_P, CFG_M, DT, noise=False)
    fd = calibrate.forward_difference(traj, DT).dropna(subset=["a_fd"]).join(runs["task"], on="run_id")
    p = {k: fd["task"].map(params[k]).to_numpy() for k in params.columns}
    a_idm = idm.accel(fd["s"].to_numpy(), fd["v_f"].to_numpy(), fd["v_l"].to_numpy(), p)
    moving = (fd["v_f"].to_numpy() + a_idm * DT) > 0          # no clipping at standstill
    assert np.allclose(fd["a_fd"].to_numpy()[moving], a_idm[moving], atol=1e-8)


def test_some_runs_stop_and_records_are_smooth():
    runs, traj, params = synthetic.make_dataset(CFG_P, CFG_M, DT)
    stopped = traj.groupby("run_id")["v_l"].min() < 0.5
    assert stopped.sum() > 0
    a = calibrate.forward_difference(traj, DT)["a_fd"].dropna()
    # unsmoothed speed noise of 0.05 m/s alone would give an std of about 0.7 m/s2
    assert a.std() < 0.6


def test_existing_pipeline_accepts_the_synthetic_data():
    runs, traj, params = synthetic.make_dataset(CFG_P, CFG_M, DT)
    traj_fd = calibrate.forward_difference(traj, DT)
    seg = sim_calibrate.make_segments(traj_fd, runs, runs.index, CFG_M, DT)
    assert len(seg["length"]) == 2 * runs["task"].nunique()
    true = params.iloc[0].to_dict()
    task = params.index[0]
    one = sim_calibrate.make_segments(traj_fd, runs, runs.index[runs["task"] == task], CFG_M, DT)
    rmse, failed = sim_calibrate.simulate_idm_batch(true, one, DT)
    assert not failed.any() and rmse.max() < 2.0          # true parameters fit (noise only)
