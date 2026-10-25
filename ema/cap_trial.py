"""Validation-only trials for the fine-tuning settings of m4a_finetuned (F026, F028).

A trial fine-tunes the pretrained network exactly as m4a_finetuned does (same seeds,
episodes and settings), except for the step cap, optionally early stopping and the
learning rate, which are taken from a section of config/m4.yaml (cap_trial, lr_trial).
Only validation curves are kept; no test run is simulated. choose_cap applies the rule
fixed before the cap trial: the new cap is the smallest multiple of `grid` steps at
which, in every trial fold, the best validation loss so far is within `tol` of the best
validation loss over the whole trial. instability and lr_decision apply the rules fixed
before the learning-rate trial (F028; the instability thresholds were fixed with the code).
"""
import hashlib
import json
import subprocess

import numpy as np
import pandas as pd

from ema import config, methods

CONFIG_FILES = ["data.yaml", "evaluation.yaml", "models.yaml", "pretraining.yaml", "m4.yaml"]


def trial_config(cfg_m4, section="cap_trial"):
    """Copy of config/m4.yaml with the settings of a trial section applied to its
    finetuning part (step cap; early stopping off if early_stopping is false; lr if given)."""
    ct = cfg_m4[section]
    ft = dict(cfg_m4["finetuning"])
    ft["max_steps"] = int(ct["max_steps"])
    if "lr" in ct:
        ft["lr"] = float(ct["lr"])
    if not ct.get("early_stopping", True):
        ft["patience"] = ft["max_steps"] // int(ft["val_every"]) + 1    # never reached
    return {**cfg_m4, "finetuning": ft}


def run_fold(fold_id, rt, assignment, traj_fd, cfg_models, cfg_eval, cfg_m4_trial, cfg_pre,
             log=print):
    """Fine-tune on the fold's training runs (validation episodes from its validation runs,
    as in m4a_finetuned). Returns (history, info, training run ids)."""
    a = assignment[assignment["fold_id"] == fold_id]
    train_ids = a.loc[a["role"] == "train", "run_id"].tolist()
    if not train_ids:
        raise ValueError(f"no training runs for fold {fold_id}")
    m = methods.M4Finetuned(cfg_models, cfg_eval, cfg_m4_trial, cfg_pre, log=log)
    m.fit(traj_fd, rt, train_ids)
    return m.history, m.info, train_ids


def best_so_far(history):
    """Series: step -> best validation loss up to that step."""
    h = pd.DataFrame(history).sort_values("step")
    return pd.Series(h["val_loss"].cummin().to_numpy(), index=h["step"].to_numpy())


def choose_cap(histories, max_steps, grid=500, tol=0.01):
    """Smallest multiple of grid (below max_steps) at which every history's best validation
    loss so far is within tol of its best over the whole run; None if there is none (the
    curves still improve by more than tol near the end of the trial)."""
    curves = [best_so_far(h) for h in histories]
    for s in range(int(grid), int(max_steps), int(grid)):
        if all(c[c.index <= s].iloc[-1] <= c.iloc[-1] * (1 + tol) for c in curves):
            return s
    return None


def instability(history, start_step=500, rel=0.10, run=3):
    """Reason why a trial curve counts as unstable, or None: a non-finite validation loss,
    a skipped step, or, after start_step, `run` consecutive validation losses more than
    `rel` above the best validation loss so far."""
    h = pd.DataFrame(history).sort_values("step")
    if not np.isfinite(h["val_loss"]).all():
        return "non-finite validation loss"
    if (h["skipped"].fillna(0) > 0).any():
        return "skipped steps"
    best = h["val_loss"].cummin()
    high = (h["val_loss"] > best * (1 + rel)) & (h["step"] > start_step)
    count = 0
    for flag, step in zip(high, h["step"]):
        count = count + 1 if flag else 0
        if count >= run:
            return f"validation loss more than {rel:.0%} above its best for {run} evaluations (step {step})"
    return None


def lr_decision(new_histories, ref_histories, max_steps, tol=0.01):
    """Rule fixed before the learning-rate trial (F028): adopt the new learning rate if no
    trial fold is unstable and each fold's best validation loss is at most (1 + tol) times
    its best in the reference trial; the cap then follows from choose_cap on the new
    curves. Returns (adopt, cap, per-fold rows)."""
    rows = []
    for new, ref in zip(new_histories, ref_histories):
        best_new = float(best_so_far(new).iloc[-1])
        best_ref = float(best_so_far(ref).iloc[-1])
        reason = instability(new)
        rows.append(dict(best_new=best_new, best_ref=best_ref, limit=best_ref * (1 + tol),
                         unstable=reason, ok=reason is None and best_new <= best_ref * (1 + tol)))
    adopt = all(r["ok"] for r in rows)
    cap = choose_cap(new_histories, max_steps, tol=tol) if adopt else None
    return adopt, cap, rows


def check_reference(directory, commit, lr, max_steps):
    """Saved history of a reference trial fold, after checking where it comes from."""
    info = json.loads((directory / "run_info.json").read_text(encoding="utf-8"))
    problems = []
    if not str(info.get("git_commit", "")).startswith(commit) or info.get("git_dirty"):
        problems.append(f"commit {info.get('git_commit')} dirty={info.get('git_dirty')}")
    if float(info["finetuning"]["lr"]) != float(lr):
        problems.append(f"lr {info['finetuning']['lr']}")
    if int(info["finetuning"]["max_steps"]) != int(max_steps):
        problems.append(f"max_steps {info['finetuning']['max_steps']}")
    if problems:
        raise ValueError(f"reference {directory} is not the expected trial: " + "; ".join(problems))
    return load_history(directory)


def git_state(root=None):
    root = root or config.REPO_ROOT
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=root).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                           cwd=root).stdout.strip() != ""
    return head, dirty


def config_md5():
    h = hashlib.md5()
    for name in CONFIG_FILES:
        h.update((config.CONFIG_DIR / name).read_bytes())
    return h.hexdigest()


def load_history(directory):
    """Saved history of one trial fold, as a list of dicts."""
    return pd.read_csv(directory / "history.csv").to_dict("records")


def save_fold(directory, history, info):
    """history.csv and run_info.json of one trial fold."""
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
    (directory / "run_info.json").write_text(
        json.dumps(info, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
        encoding="utf-8")
