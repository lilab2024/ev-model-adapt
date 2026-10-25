"""Validation-only trial for the fine-tuning step cap of m4a_finetuned (F026, option C).

The trial fine-tunes the pretrained network exactly as m4a_finetuned does (same seeds,
episodes and settings), except for the step cap and, optionally, early stopping, which are
taken from config/m4.yaml cap_trial. Only validation curves are kept; no test run is
simulated. choose_cap applies the rule fixed before the trial (D022): the new cap is the
smallest multiple of `grid` steps at which, in every trial fold, the best validation loss
so far is within `tol` of the best validation loss over the whole trial.
"""
import hashlib
import json
import subprocess

import pandas as pd

from ema import config, methods

CONFIG_FILES = ["data.yaml", "evaluation.yaml", "models.yaml", "pretraining.yaml", "m4.yaml"]


def trial_config(cfg_m4):
    """Copy of config/m4.yaml with the cap-trial settings applied to its finetuning part."""
    ct = cfg_m4["cap_trial"]
    ft = dict(cfg_m4["finetuning"])
    ft["max_steps"] = int(ct["max_steps"])
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
