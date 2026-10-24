import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from ema import calibrate, sim_calibrate, simulate, torch_sim
from ema.models import idm

DT = 0.1
P = dict(v0=33.0, T=1.5, s0=2.0, a=1.0, b=1.5, delta=4.0)
CFG_M = {"sim_calibration": {"segment_s": 30.0, "segments_per_task": 4, "seed": 0}}


def segments():
    """IDM-driven runs behind a leader with a speed dip, cut into 30 s segments."""
    t = np.arange(400) * DT
    trajs, tasks = [], {}
    for k in range(4):
        rid = f"r{k}"
        v_lead = 20.0 - 6.0 * (1 + 0.2 * k) * np.exp(-((t - 15.0) / 5.0) ** 2)
        sim = simulate.rollout(idm.IDMController(P), v_lead, s0=40.0, v0=20.0, dt=DT)
        noise = np.random.default_rng(k).normal(0.0, 0.3, len(t))
        trajs.append(pd.DataFrame({"run_id": rid, "t": t, "v_f": sim["v"], "v_l": v_lead,
                                   "s": sim["s"] + noise}))
        tasks[rid] = "A" if k < 2 else "B"
    traj_fd = calibrate.forward_difference(pd.concat(trajs, ignore_index=True), DT)
    rt = pd.DataFrame({"task": pd.Series(tasks)}).rename_axis("run_id")
    return sim_calibrate.make_segments(traj_fd, rt, rt.index, CFG_M, DT)


def torch_params(p, n, dtype=torch.float64, requires_grad=False):
    out = {}
    for k, v in p.items():
        t = torch.full((n,), float(v), dtype=dtype)
        if requires_grad and k != "delta":
            t.requires_grad_(True)
        out[k] = t
    return out


def test_torch_rollout_matches_numpy_simulation():
    seg = segments()
    p2 = dict(v0=40.0, T=1.2, s0=3.0, a=1.4, b=2.0, delta=4.0)
    rmse_np, failed = sim_calibrate.simulate_idm_batch(p2, seg, DT)
    assert not failed.any()
    tseg = torch_sim.to_torch(seg, dtype=torch.float64)
    out = torch_sim.rollout(torch_params(p2, len(seg["length"])), tseg, DT)
    rmse_t = torch_sim.spacing_rmse(out["s"], tseg, eps=0.0).numpy()
    assert np.allclose(rmse_t, rmse_np, rtol=1e-10, atol=1e-10)
    assert not out["collided"].any()


def test_gradient_matches_finite_difference():
    seg = segments()
    tseg = torch_sim.to_torch(seg, dtype=torch.float64)
    n = len(seg["length"])
    p = torch_params(dict(v0=40.0, T=1.2, s0=3.0, a=1.4, b=2.0, delta=4.0), n, requires_grad=True)
    loss = torch_sim.spacing_rmse(torch_sim.rollout(p, tseg, DT)["s"], tseg).mean()
    loss.backward()
    for k in ["T", "a", "s0"]:
        h = 1e-6
        vals = []
        for sign in (1, -1):
            q = {kk: vv.detach().clone() for kk, vv in p.items()}
            q[k] = q[k] + sign * h
            vals.append(torch_sim.spacing_rmse(torch_sim.rollout(q, tseg, DT)["s"], tseg).mean().item())
        fd = (vals[0] - vals[1]) / (2 * h)
        assert p[k].grad.sum().item() == pytest.approx(fd, rel=1e-4, abs=1e-8)


def test_collision_is_flagged():
    n = 50
    s_obs = np.linspace(5.0, -1.0, n)                 # gap closes while the follower stands
    seg = dict(v_lead=np.zeros((1, n)), s_obs=s_obs[None, :],
               lead_pos=simulate.leader_position(s_obs, np.zeros(n), DT)[None, :],
               length=np.array([n]), s0=np.array([5.0]), v0=np.array([0.0]))
    out = torch_sim.rollout(torch_params(P, 1), torch_sim.to_torch(seg, torch.float64), DT)
    assert out["collided"][0]


def test_gradient_descent_recovers_a_parameter():
    seg = segments()
    tseg = torch_sim.to_torch(seg, dtype=torch.float64)
    n = len(seg["length"])
    log_T = torch.tensor(np.log(2.5), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([log_T], lr=0.05)
    fixed = torch_params(P, n)
    for _ in range(150):
        opt.zero_grad()
        p = {**fixed, "T": torch.exp(log_T).expand(n)}
        loss = torch_sim.spacing_rmse(torch_sim.rollout(p, tseg, DT)["s"], tseg).mean()
        loss.backward()
        opt.step()
    assert float(torch.exp(log_T).detach()) == pytest.approx(P["T"], rel=0.05)
