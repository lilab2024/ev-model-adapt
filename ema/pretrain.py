"""Pretraining of the M4 network on synthetic IDM vehicles (D017) and its diagnostics.

Synthetic vehicles are split into training and validation vehicles. The network is
trained on episodes of the training vehicles; a fixed set of validation episodes
controls early stopping. Diagnostics on the validation vehicles, per context length:
closed-loop spacing RMSE of the targets with the predicted parameters, with the true
parameters (lower bound) and with the population median of the training vehicles (no
adaptation), and the Spearman correlation of predicted vs true parameters.
"""
import datetime
import hashlib
import json
import subprocess

import numpy as np
import pandas as pd
import torch
from scipy import stats

from ema import config, nets, synthetic, train_m4


def split_vehicles(tasks, val_share, seed):
    """Disjoint (train, val) lists of tasks; val = round(val_share * n), at least 1."""
    tasks = sorted(tasks)
    rng = np.random.default_rng(int(seed))
    n_val = max(1, int(round(float(val_share) * len(tasks))))
    val = set(rng.choice(tasks, size=n_val, replace=False))
    return [t for t in tasks if t not in val], sorted(val)


def fixed_episodes(arrays, task_runs, tasks, lengths, cfg_tr, dt, seed):
    """One episode per (task, context length), reproducible. Returns a list of
    (task, length, episode)."""
    sampler = train_m4.EpisodeSampler(arrays, {t: task_runs[t] for t in tasks}, cfg_tr, dt, seed)
    return [(t, float(L), sampler.one(t, float(L))) for t in sampler.tasks for L in lengths]


def _as_tensors(param_rows, names, device="cpu"):
    return {k: torch.tensor([float(r[k]) for r in param_rows], device=device) for k in names}


def diagnostics(net, items, params_true, pop, dt, scales, min_gap=0.1):
    """DataFrame per context length: mean RMSE with predicted / true / population
    parameters and Spearman correlations of predicted vs true parameters."""
    names = list(params_true.columns)
    episodes = [e for _, _, e in items]
    device = train_m4.device_of(net)
    x, lengths, ep = train_m4.context_tensors(episodes, dt, scales, device)
    net.eval()
    with torch.no_grad():
        p_pred = net(x, lengths, ep, len(episodes))
        truth = [params_true.loc[t] for t, _, _ in items]
        p_true = _as_tensors(truth, names, device)
        p_pop = _as_tensors([pop] * len(items), names, device)
        r_pred = train_m4.targets_rmse(p_pred, episodes, dt, min_gap).cpu().numpy()
        r_true = train_m4.targets_rmse(p_true, episodes, dt, min_gap).cpu().numpy()
        r_pop = train_m4.targets_rmse(p_pop, episodes, dt, min_gap).cpu().numpy()
        p_pred = {k: v.cpu() for k, v in p_pred.items()}
    df = pd.DataFrame({"length_s": [L for _, L, _ in items], "rmse_pred": r_pred,
                       "rmse_true": r_true, "rmse_pop": r_pop})
    for k in names:
        if params_true[k].nunique() > 1:
            df[f"pred_{k}"] = p_pred[k].numpy()
            df[f"true_{k}"] = [float(r[k]) for r in truth]
    rows = []
    for L, g in df.groupby("length_s"):
        row = {"length_s": L, "n": len(g), "rmse_pred": g["rmse_pred"].mean(),
               "rmse_true": g["rmse_true"].mean(), "rmse_pop": g["rmse_pop"].mean()}
        for k in names:
            if f"pred_{k}" in g:
                row[f"spearman_{k}"] = stats.spearmanr(g[f"pred_{k}"], g[f"true_{k}"]).statistic
        rows.append(row)
    return pd.DataFrame(rows), df


def provenance():
    root = config.REPO_ROOT
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=root).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                           cwd=root).stdout.strip() != ""
    h = hashlib.md5()
    for name in ["evaluation.yaml", "models.yaml", "pretraining.yaml"]:
        h.update((config.CONFIG_DIR / name).read_bytes())
    return dict(git_commit=head, git_dirty=dirty, config_md5=h.hexdigest())


def run(out_dir, cfg_pre=None, cfg_models=None, dt=0.1, log=print, n_vehicles=None,
        resume_from=None):
    """Generate the synthetic set, pretrain (optionally starting from saved weights),
    write net.pt, history.csv, diagnostics.csv, diagnostics_episodes.csv and
    run_info.json to out_dir. Returns the diagnostics table."""
    cfg_pre = cfg_pre or config.load_yaml("pretraining.yaml")
    cfg_models = cfg_models or config.load_yaml("models.yaml")
    tr, nw = cfg_pre["training"], cfg_pre["network"]
    t0 = datetime.datetime.now()
    runs, traj, params = synthetic.make_dataset(cfg_pre, cfg_models, dt, n_vehicles=n_vehicles)
    arrays = train_m4.run_arrays(traj)
    task_runs = {t: list(r) for t, r in runs.groupby("task").groups.items()}
    train_tasks, val_tasks = split_vehicles(task_runs, tr["val_share"], tr["seed"])
    log(f"synthetic: {len(task_runs)} vehicles ({len(train_tasks)} train, {len(val_tasks)} val), "
        f"{len(runs)} runs")

    device = train_m4.resolve_device(tr.get("device", "cpu"))
    torch.manual_seed(int(tr["seed"]))
    net = nets.AdaptNet(nw, cfg_models["idm"]["params"])
    if resume_from is not None:
        net.load_state_dict(torch.load(resume_from, map_location="cpu"))
        log(f"starting from saved weights: {resume_from}")
        if "resume_lr" in tr:
            tr = {**tr, "lr": tr["resume_lr"]}
    log(f"learning rate: {tr['lr']}")
    net = net.to(device)
    log(f"device: {device}")
    sampler = train_m4.EpisodeSampler(arrays, {t: task_runs[t] for t in train_tasks}, tr, dt,
                                      int(tr["seed"]))
    val_sampler = train_m4.EpisodeSampler(arrays, {t: task_runs[t] for t in val_tasks}, tr, dt,
                                          int(tr["seed"]) + 1)
    val_eps = val_sampler.sample(int(tr["val_episodes"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    history = train_m4.train(net, sampler, val_eps, tr, dt, nw["scales"], log=log,
                             min_gap=cfg_models["idm"]["min_gap_m"], checkpoint_dir=out_dir)

    pop = {k: float(np.exp(np.log(params.loc[train_tasks, k]).median())) for k in params.columns}
    items = fixed_episodes(arrays, task_runs, val_tasks, tr["context_s"], tr, dt, int(tr["seed"]) + 2)
    summary, per_episode = diagnostics(net, items, params, pop, dt, nw["scales"],
                                       cfg_models["idm"]["min_gap_m"])

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in net.state_dict().items()}, out_dir / "net.pt")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    summary.to_csv(out_dir / "diagnostics.csv", index=False)
    per_episode.to_csv(out_dir / "diagnostics_episodes.csv", index=False)
    info = dict(**provenance(), device=str(device), lr=float(tr["lr"]),
                n_skipped_steps=history[-1]["skipped"] if history else 0,
                resumed_from=str(resume_from) if resume_from is not None else None,
                n_vehicles=len(task_runs), n_train=len(train_tasks),
                n_val=len(val_tasks), steps_run=history[-1]["step"] if history else 0,
                best_val_loss=min(h["val_loss"] for h in history) if history else None,
                population_params=pop, started=t0.isoformat(timespec="seconds"),
                finished=datetime.datetime.now().isoformat(timespec="seconds"))
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return summary
