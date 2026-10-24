"""Episodes and training loop for the M4 network (D017).

An episode = one task (vehicle): a context cut from some of its runs and target
segments from other runs of the same task. The network turns the context into IDM
parameters; the loss is the closed-loop spacing RMSE of the targets, simulated with the
differentiable rollout of ema.torch_sim (same rules as the evaluation).
"""
import copy
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ema import nets, simulate, torch_sim


def run_arrays(traj, run_ids=None):
    """dict run_id -> dict(s, v_f, v_l) numpy arrays (sorted by time)."""
    d = traj if run_ids is None else traj[traj["run_id"].isin(set(run_ids))]
    return {rid: {c: g[c].to_numpy(float) for c in ("s", "v_f", "v_l")}
            for rid, g in d.sort_values(["run_id", "t"]).groupby("run_id", sort=False)}


class EpisodeSampler:
    """Random episodes from tasks with at least two runs; context and target runs are
    disjoint. Context: random start in the first context run, consecutive runs until the
    requested length (capped so that at least one run is left for the targets)."""

    def __init__(self, arrays, task_runs, cfg_tr, dt, seed):
        self.arrays, self.dt, self.cfg = arrays, dt, cfg_tr
        self.tasks = sorted(t for t, r in task_runs.items() if len(r) >= 2)
        self.task_runs = {t: sorted(task_runs[t]) for t in self.tasks}
        self.rng = np.random.default_rng(seed)

    def one(self, task, length_s):
        rng, dt = self.rng, self.dt
        rids = list(self.task_runs[task])
        rng.shuffle(rids)
        need = int(round(length_s / dt)) + 1
        ctx_pieces, used = [], 0
        for i, rid in enumerate(rids[:-1]):                   # keep >= 1 run for the targets
            a = self.arrays[rid]
            n = len(a["s"])
            start = int(rng.integers(0, max(n - need, 0) + 1)) if i == 0 else 0
            stop = min(n, start + need)
            ctx_pieces.append({k: v[start:stop] for k, v in a.items()})
            used = i + 1
            need -= stop - start - 1
            if need <= 1:
                break
        tgt_len = int(round(self.cfg["target_s"] / dt)) + 1
        targets = []
        for rid in rids[used:used + int(self.cfg["targets_per_episode"])]:
            a = self.arrays[rid]
            n = len(a["s"])
            start = int(rng.integers(0, max(n - tgt_len, 0) + 1))
            targets.append({k: v[start:start + tgt_len] for k, v in a.items()})
        return ctx_pieces, targets

    def sample(self, n_episodes):
        tasks = self.rng.choice(self.tasks, n_episodes)
        lengths = self.rng.choice(self.cfg["context_s"], n_episodes)
        return [self.one(t, float(L)) for t, L in zip(tasks, lengths)]


def device_of(net):
    return next(net.parameters()).device


def resolve_device(name):
    """torch.device for 'cuda' or 'cpu'; a clear error if a GPU is requested but absent."""
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("device: cuda requested, but no GPU is available "
                           "(Colab: Runtime > Change runtime type > GPU)")
    return torch.device(name)


def pack_targets(episodes, dt, dtype=torch.float32, device="cpu"):
    """Targets of a list of episodes as a torch segment dict (+ episode index)."""
    tg = [(i, t) for i, (_, targets) in enumerate(episodes) for t in targets]
    lengths = np.array([len(t["s"]) for _, t in tg])
    T = int(lengths.max())
    v_lead, s_obs, lead_pos = (np.zeros((len(tg), T)) for _ in range(3))
    for j, (_, t) in enumerate(tg):
        n = len(t["s"])
        lp = simulate.leader_position(t["s"], t["v_f"], dt)
        v_lead[j, :n], s_obs[j, :n], lead_pos[j, :n] = t["v_l"], t["s"], lp
        v_lead[j, n:], s_obs[j, n:], lead_pos[j, n:] = t["v_l"][-1], t["s"][-1], lp[-1]
    seg = dict(v_lead=v_lead, s_obs=s_obs, lead_pos=lead_pos, length=lengths,
               s0=s_obs[:, 0].copy(), v0=np.array([t["v_f"][0] for _, t in tg]))
    return (torch_sim.to_torch(seg, dtype=dtype, device=device),
            torch.tensor([i for i, _ in tg], device=device))


def context_tensors(episodes, dt, scales, device="cpu"):
    feats = [[nets.piece_features(p["s"], p["v_f"], p["v_l"], dt, scales) for p in ctx]
             for ctx, _ in episodes]
    return nets.collate_contexts(feats, device)


def targets_rmse(p, episodes, dt, min_gap=0.1, dtype=torch.float32):
    """Per-episode mean closed-loop spacing RMSE of the targets for IDM parameters p
    (dict of tensors of shape (n_episodes,), all on one device)."""
    device = next(iter(p.values())).device
    seg, tgt_ep = pack_targets(episodes, dt, dtype=dtype, device=device)
    out = torch_sim.rollout({k: v[tgt_ep] for k, v in p.items()}, seg, dt, min_gap=min_gap)
    rmse = torch_sim.spacing_rmse(out["s"], seg)
    per_ep = torch.zeros(len(episodes), dtype=rmse.dtype, device=device).index_add(0, tgt_ep, rmse)
    return per_ep / torch.bincount(tgt_ep, minlength=len(episodes)).clamp(min=1)


def episode_loss(net, episodes, dt, scales, min_gap=0.1):
    """Mean over episodes of the mean target spacing RMSE (closed loop)."""
    x, lengths, ep = context_tensors(episodes, dt, scales, device_of(net))
    p = net(x, lengths, ep, len(episodes))
    return targets_rmse(p, episodes, dt, min_gap, dtype=x.dtype).mean()


def _atomic_write(path, write):
    """Write to a temporary file, then replace the target (a crash never leaves a half file)."""
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def save_checkpoint(directory, best_state, history, step, best_val, best_step):
    """net_best.pt (only if best_state is given), history.csv and checkpoint.json."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        _atomic_write(directory / "net_best.pt", lambda p: torch.save(
            {k: v.detach().cpu() for k, v in best_state.items()}, p))
    _atomic_write(directory / "history.csv",
                  lambda p: pd.DataFrame(history).to_csv(p, index=False))
    info = dict(step=step, best_step=best_step, best_val_loss=best_val,
                saved=datetime.datetime.now().isoformat(timespec="seconds"))
    _atomic_write(directory / "checkpoint.json",
                  lambda p: p.write_text(json.dumps(info, indent=2), encoding="utf-8"))


def train(net, sampler, val_episodes, cfg_tr, dt, scales, log=print, min_gap=0.1,
          checkpoint_dir=None):
    """Adam with gradient clipping; validation loss every val_every steps; the best state
    (lowest validation loss) is restored; stops after `patience` evaluations without
    improvement. With checkpoint_dir, the best weights (net_best.pt), the history and
    checkpoint.json are written after every validation, so an interrupted run can be
    resumed from net_best.pt. Returns the history as a list of dicts."""
    torch.manual_seed(int(cfg_tr["seed"]))
    opt = torch.optim.Adam(net.parameters(), lr=float(cfg_tr["lr"]))
    best, best_state, bad, history = float("inf"), copy.deepcopy(net.state_dict()), 0, []
    t_start = time.time()
    best_step = 0
    for step in range(1, int(cfg_tr["steps"]) + 1):
        net.train()
        loss = episode_loss(net, sampler.sample(int(cfg_tr["batch_episodes"])), dt, scales, min_gap)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), float(cfg_tr["grad_clip"]))
        opt.step()
        if step % int(cfg_tr["val_every"]) == 0:
            net.eval()
            with torch.no_grad():
                vl = float(episode_loss(net, val_episodes, dt, scales, min_gap))
            tl = float(loss.detach())
            elapsed = time.time() - t_start
            history.append(dict(step=step, train_loss=tl, val_loss=vl, elapsed_s=elapsed))
            log(f"step {step:5d}  train {tl:7.3f}  val {vl:7.3f}  "
                f"({elapsed / step:.2f} s/step, {elapsed / 60:.1f} min)")
            improved = vl < best - 1e-4
            if improved:
                best, best_state, bad, best_step = vl, copy.deepcopy(net.state_dict()), 0, step
            else:
                bad += 1
            if checkpoint_dir is not None:
                save_checkpoint(checkpoint_dir, best_state if improved else None, history,
                                step, best, best_step)
            if bad >= int(cfg_tr["patience"]):
                break
    net.load_state_dict(best_state)
    return history


def predict_params(net, pieces, dt, scales):
    """IDM parameters (dict of floats) for one context given as a list of DataFrames or
    dicts with s, v_f, v_l."""
    feats = [nets.piece_features(np.asarray(p["s"]), np.asarray(p["v_f"]), np.asarray(p["v_l"]),
                                 dt, scales) for p in pieces]
    x, lengths, ep = nets.collate_contexts([feats], device_of(net))
    net.eval()
    with torch.no_grad():
        p = net(x, lengths, ep, 1)
    return {k: float(v[0]) for k, v in p.items()}
