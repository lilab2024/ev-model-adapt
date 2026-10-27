import numpy as np
import pandas as pd
import pytest

from ema import calibrate, context, evaluate, methods, protocol, simulate
from ema.models import idm
from test_population_prior import BASE, CFG_E, CFG_M, DT

VEHICLES = [("EV", "A", 0.8), ("EV", "B", 0.7), ("ICE", "C", 1.0), ("ICE", "D", 1.2)]


def setup(levels):
    """EV A and B, ICE C and D (two gaps each, 4 configurations x 1 run of 60 s)."""
    runs, trajs = [], []
    t = np.arange(600) * DT
    for pt, veh, f in VEHICLES:
        p = {**BASE, "T": BASE["T"] * f, "a": BASE["a"] / f}
        for gap in ["g1", "g2"]:
            for k in range(4):
                rid = f"{pt}_{veh}_{gap}_{k}"
                v_lead = 20.0 - 6.0 * (1 + 0.2 * k) * np.exp(-((t - 25.0) / 6.0) ** 2)
                sim = simulate.rollout(idm.IDMController(p), v_lead, s0=40.0, v0=20.0, dt=DT)
                trajs.append(pd.DataFrame({"run_id": rid, "t": t, "v_f": sim["v"],
                                           "v_l": v_lead, "s": sim["s"]}))
                runs.append(dict(run_id=rid, powertrain=pt, vehicle=veh, gap=gap, profile="dip",
                                 desired_mph=0.0, v_free_mph=45.0, v_reduced_mph=20.0 + 5 * k,
                                 qc_pass=True, exclude=False, duration_s=float(t[-1])))
    cfg_e = dict(CFG_E, levels=levels)
    rt = protocol.run_table(pd.DataFrame(runs), cfg_e)
    folds = protocol.make_folds(rt, cfg_e)
    asg = protocol.assign_runs(rt, folds, cfg_e)
    ct = context.context_table(rt, cfg_e)
    return cfg_e, rt, folds, asg, ct, calibrate.forward_difference(pd.concat(trajs), DT)


def test_m2b_uses_only_training_tasks_of_the_test_powertrain():
    methods.PopulationPrior._task_fits.clear()
    cfg_e, rt, folds, asg, ct, traj_fd = setup(["L2_new_vehicle"])
    res, st = evaluate.run_evaluation(methods.PopulationPriorSamePowertrain, CFG_M, rt, folds,
                                      asg, ct, traj_fd, cfg_e, verbose=False)
    for f, s in zip(folds, st):
        pt = f["test"][0].split(" | ")[0]
        same = [t for t in f["train"] if t.split(" | ")[0] == pt]
        assert s["method"] == "population_prior_same_powertrain" and s["prior_powertrain"] == pt
        assert s["n_train_tasks"] == len(same) == 2
        assert set(filter(None, s["pseudo_tasks"].split("; "))) <= set(same)
    assert not res["collided"].any()


def test_m2b_at_l2_equals_m2_at_l2_no_ice_for_ev_vehicles():
    methods.PopulationPrior._task_fits.clear()
    cfg_e, rt, folds, asg, ct, traj_fd = setup(["L2_new_vehicle", "L2_no_ice"])
    l2_ev = [f for f in folds if f["level"] == "L2_new_vehicle" and f["test"][0].startswith("EV")]
    no_ice = [f for f in folds if f["level"] == "L2_no_ice"]
    m2b, _ = evaluate.run_evaluation(methods.PopulationPriorSamePowertrain, CFG_M, rt, l2_ev, asg,
                                     ct, traj_fd, cfg_e, verbose=False)
    m2, _ = evaluate.run_evaluation(methods.PopulationPrior, CFG_M, rt, no_ice, asg, ct, traj_fd,
                                    cfg_e, verbose=False)
    assert len(l2_ev) == len(no_ice) == 2 and len(m2b) == len(m2) > 0
    keys = ["task", "draw", "length_s", "run_id"]
    a = m2b.sort_values(keys).reset_index(drop=True)
    b = m2.sort_values(keys).reset_index(drop=True)
    for col in ["spacing_rmse", "ctx_idm_T", "ctx_lambda"]:
        assert np.array_equal(a[col].to_numpy(), b[col].to_numpy())


def test_m2b_refuses_other_levels():
    cfg_e, rt, folds, asg, ct, traj_fd = setup(["L3_cross_powertrain"])
    with pytest.raises(ValueError, match="only at the levels"):
        evaluate.run_evaluation(methods.PopulationPriorSamePowertrain, CFG_M, rt, folds, asg, ct,
                                traj_fd, cfg_e, verbose=False)
