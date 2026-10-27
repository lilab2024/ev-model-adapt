"""m2b_residual (D027): IDM plus a bounded learned acceleration correction.

acceleration = a_IDM + r, r = a_max * tanh(MLP(features)), features (scaled):
gap, follower speed, speed difference (leader - follower), IDM acceleration, leader
acceleration ((v_lead(k) - v_lead(k-1)) / dt, 0 at the first step) and the follower's
acceleration of the previous step (0 at the first step). The last layer starts at zero,
so an untrained correction is exactly the IDM. The same rules are used in the
differentiable training rollout (PyTorch) and in the controller for the evaluation
(NumPy, CPU), with the Euler and leader-position simulation of ema.simulate.
"""
import math

import numpy as np
import torch

from ema import simulate, torch_sim
from ema.models import idm

FEATURES = ("gap", "v", "dv", "a_idm", "a_lead", "a_prev")


def scale_vector(cfg_res):
    return np.array([float(cfg_res["scales"][k]) for k in FEATURES])


class ResidualNet(torch.nn.Module):
    """MLP with tanh layers; output a_max * tanh(.); the last layer is initialised to zero."""

    def __init__(self, hidden=(64, 64), a_max=0.5):
        super().__init__()
        layers, n_in = [], len(FEATURES)
        for h in hidden:
            layers += [torch.nn.Linear(n_in, int(h)), torch.nn.Tanh()]
            n_in = int(h)
        self.body = torch.nn.Sequential(*layers)
        self.out = torch.nn.Linear(n_in, 1)
        torch.nn.init.zeros_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)
        self.a_max = float(a_max)

    def forward(self, x):
        return self.a_max * torch.tanh(self.out(self.body(x))).squeeze(-1)


def numpy_weights(net):
    """[(W, b), ...] as float64 arrays, in layer order (the last pair is the output layer)."""
    lin = [m for m in net.body if isinstance(m, torch.nn.Linear)] + [net.out]
    return [(m.weight.detach().cpu().double().numpy(), m.bias.detach().cpu().double().numpy())
            for m in lin]


def numpy_forward(weights, a_max, x):
    """The network on a (n, 6) float64 array of scaled features; returns shape (n,)."""
    h = np.asarray(x, dtype=float)
    for W, b in weights[:-1]:
        h = np.tanh(h @ W.T + b)
    W, b = weights[-1]
    return a_max * np.tanh((h @ W.T + b)[:, 0])


class ResidualIDMController:
    """IDM + correction as a closed-loop controller (interface of IDMController)."""

    def __init__(self, params, weights, a_max, scales, dt, min_gap=0.1):
        self.idm = idm.IDMController(params, min_gap)
        self.weights, self.a_max = weights, float(a_max)
        self.scales, self.dt = np.asarray(scales, dtype=float), float(dt)
        self.prev_v_lead, self.a_prev = None, 0.0

    def reset(self, s0, v0, v_lead0):
        self.prev_v_lead, self.a_prev = None, 0.0

    def accel(self, s, v, v_lead):
        a_idm = self.idm.accel(s, v, v_lead)
        a_lead = 0.0 if self.prev_v_lead is None else (v_lead - self.prev_v_lead) / self.dt
        x = np.array([[s, v, v_lead - v, a_idm, a_lead, self.a_prev]]) / self.scales
        a = a_idm + float(numpy_forward(self.weights, self.a_max, x)[0])
        self.prev_v_lead, self.a_prev = v_lead, a
        return a


def rollout(net, p, seg, dt, scales, min_gap=0.1):
    """Differentiable rollout of IDM + correction on a batch of windows.

    seg: tensors v_lead, lead_pos, s_obs (n x T), s0, v0, length (n); p: IDM parameters,
    tensors of shape (n,). Returns (spacing RMSE per window, mean squared correction per
    window, simulated gaps n x T); only steps before length count.
    """
    v_lead, lead_pos, length = seg["v_lead"], seg["lead_pos"], seg["length"]
    n, T = v_lead.shape
    sc = torch.as_tensor(scales, dtype=v_lead.dtype, device=v_lead.device)
    s, v = seg["s0"].clone(), seg["v0"].clone()
    x = lead_pos[:, 0] - s
    a_prev = torch.zeros_like(s)
    se, pen = torch.zeros_like(s), torch.zeros_like(s)
    s_hist = [s]
    for k in range(T - 1):
        a_idm = torch_sim.idm_accel(s, v, v_lead[:, k], p, min_gap)
        a_lead = torch.zeros_like(s) if k == 0 else (v_lead[:, k] - v_lead[:, k - 1]) / dt
        feat = torch.stack([s, v, v_lead[:, k] - v, a_idm, a_lead, a_prev], dim=1) / sc
        r = net(feat)
        a = a_idm + r
        x = x + v * dt
        v = torch.clamp(v + a * dt, min=0.0)
        s = lead_pos[:, k + 1] - x
        active = ((k + 1) < length).to(s.dtype)
        se = se + active * (s - seg["s_obs"][:, k + 1]) ** 2
        pen = pen + active * r ** 2
        a_prev = a
        s_hist.append(s)
    lf = length.to(s.dtype)
    rmse = torch.sqrt(se / lf + 1e-8)
    return rmse, pen / torch.clamp(lf - 1.0, min=1.0), torch.stack(s_hist, dim=1)


def pack_windows(windows, dt, dtype=torch.float32, device="cpu"):
    """Tensors for a list of dicts with numpy arrays s, v_f, v_l (padded with the last value)."""
    n = len(windows)
    T = max(len(w["s"]) for w in windows)
    v_lead, lead_pos, s_obs = np.zeros((n, T)), np.zeros((n, T)), np.zeros((n, T))
    for i, w in enumerate(windows):
        m = len(w["s"])
        lp = simulate.leader_position(w["s"], w["v_f"], dt)
        v_lead[i, :m], lead_pos[i, :m], s_obs[i, :m] = w["v_l"], lp, w["s"]
        v_lead[i, m:], lead_pos[i, m:], s_obs[i, m:] = w["v_l"][-1], lp[-1], w["s"][-1]
    t = lambda a: torch.as_tensor(a, dtype=dtype, device=device)
    return dict(v_lead=t(v_lead), lead_pos=t(lead_pos), s_obs=t(s_obs),
                s0=t([w["s"][0] for w in windows]), v0=t([w["v_f"][0] for w in windows]),
                length=torch.as_tensor([len(w["s"]) for w in windows], device=device))


def params_tensor(param_list, dtype=torch.float32, device="cpu"):
    """IDM parameters of every window as tensors of shape (n,)."""
    return {k: torch.as_tensor([float(p[k]) for p in param_list], dtype=dtype, device=device)
            for k in idm.PARAM_NAMES}


class WindowSampler:
    """Random windows of window_s from the runs of the given tasks: task chosen uniformly,
    then a run of the task uniformly, then a random start (shorter runs used whole)."""

    def __init__(self, arrays, task_runs, window_s, dt, seed):
        self.arrays, self.task_runs = arrays, {t: sorted(r) for t, r in task_runs.items() if r}
        self.tasks = sorted(self.task_runs)
        self.n = int(round(window_s / dt)) + 1
        self.rng = np.random.default_rng(seed)

    def sample(self, k):
        out = []
        for _ in range(int(k)):
            task = self.tasks[int(self.rng.integers(len(self.tasks)))]
            runs = self.task_runs[task]
            a = self.arrays[runs[int(self.rng.integers(len(runs)))]]
            m = len(a["s"])
            start = int(self.rng.integers(0, max(m - self.n, 0) + 1))
            out.append((task, {c: a[c][start:start + self.n] for c in ("s", "v_f", "v_l")}))
        return out


def batch_loss(net, items, task_params, dt, scales, penalty, device, min_gap=0.1):
    """Loss and its two parts for a list of (task, window): mean RMSE + penalty * mean r^2."""
    seg = pack_windows([w for _, w in items], dt, device=device)
    p = params_tensor([task_params[t] for t, _ in items], device=device)
    rmse, pen, _ = rollout(net, p, seg, dt, scales, min_gap)
    return rmse.mean() + penalty * pen.mean(), rmse.mean(), pen.mean()


def train(net, sampler, val_items, task_params, cfg_tr, dt, scales, steps, select, log=print,
          min_gap=0.1):
    """Adam with gradient clipping. With select: validation loss (the same loss on fixed
    val_items) at step 0 and every val_every steps, early stopping after patience
    evaluations without improvement, best state restored (step 0 = uncorrected IDM). Without
    select: exactly `steps` updates, final weights kept. Returns the history."""
    device = next(net.parameters()).device
    opt = torch.optim.Adam(net.parameters(), lr=float(cfg_tr["lr"]))
    pen_w = float(cfg_tr["penalty"])
    history, skipped = [], 0
    best, best_state, bad, best_step = math.inf, {k: v.clone() for k, v in net.state_dict().items()}, 0, 0

    def validate(step, train_loss):
        nonlocal best, best_state, bad, best_step
        net.eval()
        with torch.no_grad():
            vl, vr, vp = (float(x) for x in batch_loss(net, val_items, task_params, dt, scales,
                                                       pen_w, device, min_gap))
        history.append(dict(step=step, train_loss=train_loss, val_loss=vl, val_rmse=vr,
                            val_mean_r2=vp, skipped=skipped))
        log(f"step {step:5d}  train {train_loss:7.3f}  val {vl:7.3f} (rmse {vr:6.3f}, r2 {vp:.4f})"
            f"  skipped {skipped}")
        if vl < best - 1e-4:
            best, best_state, bad, best_step = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0, step
        else:
            bad += 1

    if select:
        validate(0, float("nan"))
    for step in range(1, int(steps) + 1):
        net.train()
        loss, _, _ = batch_loss(net, sampler.sample(int(cfg_tr["batch_windows"])), task_params,
                                dt, scales, pen_w, device, min_gap)
        opt.zero_grad()
        if torch.isfinite(loss):
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(net.parameters(), float(cfg_tr["grad_clip"]))
            if torch.isfinite(norm):
                opt.step()
            else:
                skipped += 1
        else:
            skipped += 1
        if select and step % int(cfg_tr["val_every"]) == 0:
            validate(step, float(loss.detach()))
            if bad >= int(cfg_tr["patience"]):
                break
        elif not select and (step % int(cfg_tr["val_every"]) == 0 or step == int(steps)):
            history.append(dict(step=step, train_loss=float(loss.detach()), skipped=skipped))
            log(f"stage 2 step {step:5d}/{steps}  train {float(loss.detach()):7.3f}  skipped {skipped}")
    if select:
        net.load_state_dict(best_state)
    net.eval()
    return history
