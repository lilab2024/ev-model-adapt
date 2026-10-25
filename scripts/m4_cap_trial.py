"""Validation-only trial for the fine-tuning step cap of m4a_finetuned (F026, option C).

Usage (from the repository root):
    python scripts/m4_cap_trial.py

For each fold in config/m4.yaml cap_trial.folds, the pretrained network is fine-tuned as
in m4a_finetuned, but up to cap_trial.max_steps and, with early_stopping false, without
early stopping. Only the validation curve is kept, in results/m4_cap_trial/<fold>/
(history.csv, run_info.json); no test run is simulated and no network is saved. A fold is
skipped if its saved run_info.json comes from the same clean commit and config files.
"""
import datetime
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ema import calibrate, cap_trial, config, data, protocol


def main():
    cfg_e, cfg_m = config.load_yaml("evaluation.yaml"), config.load_yaml("models.yaml")
    cfg_pre, cfg_m4 = config.load_yaml("pretraining.yaml"), config.load_yaml("m4.yaml")
    trial = cap_trial.trial_config(cfg_m4)
    ft, ct = trial["finetuning"], cfg_m4["cap_trial"]
    head, dirty = cap_trial.git_state()
    cfg_hash = cap_trial.config_md5()
    print(f"cap trial: folds {ct['folds']}, max steps {ft['max_steps']}, patience "
          f"{ft['patience']} (early stopping {'on' if ct.get('early_stopping', True) else 'off'}), "
          f"git {head[:7]}{' DIRTY' if dirty else ''}", flush=True)

    rt = protocol.run_table(data.load_runs(), cfg_e)
    folds = [f for f in protocol.make_folds(rt, cfg_e) if f["fold_id"] in ct["folds"]]
    missing = set(ct["folds"]) - {f["fold_id"] for f in folds}
    if missing:
        raise ValueError(f"unknown folds in cap_trial.folds: {sorted(missing)}")
    assignment = protocol.assign_runs(rt, folds, cfg_e)
    traj_fd = calibrate.forward_difference(data.load_trajectories(), cfg_e["simulation"]["dt_s"])
    out = config.REPO_ROOT / "results" / "m4_cap_trial"

    histories = []
    for f in folds:
        d = out / f["fold_id"]
        if (d / "run_info.json").exists() and not dirty:
            old = json.loads((d / "run_info.json").read_text(encoding="utf-8"))
            same = (old.get("git_commit") == head and not old.get("git_dirty")
                    and old.get("config_md5") == cfg_hash)
            if same:
                print(f"{f['fold_id']}: reused", flush=True)
                histories.append(cap_trial.load_history(d))
                continue
        print(f"{f['fold_id']}: fine-tuning", flush=True)
        t0 = time.time()
        history, info, train_ids = cap_trial.run_fold(
            f["fold_id"], rt, assignment, traj_fd, cfg_m, cfg_e, trial, cfg_pre,
            log=lambda m: print(m, flush=True))
        cap_trial.save_fold(d, history, dict(
            fold_id=f["fold_id"], git_commit=head, git_dirty=dirty, config_md5=cfg_hash,
            cap_trial=ct, finetuning=ft, n_train_runs=len(train_ids), **info,
            finished=datetime.datetime.now().isoformat(timespec="seconds"),
            runtime_s=round(time.time() - t0, 1)))
        histories.append(history)

    print()
    print(f"Saved to {out} (git {head[:7]}{' DIRTY' if dirty else ''})")
    for f, h in zip(folds, histories):
        b = cap_trial.best_so_far(h)
        marks = [s for s in range(500, int(ft["max_steps"]) + 1, 500) if s in b.index]
        print(f"{f['fold_id']}: best validation loss so far at "
              + ", ".join(f"{s}: {b[s]:.3f}" for s in marks))
    cap = cap_trial.choose_cap(histories, ft["max_steps"])
    print("rule (D022, tol 1 %, grid 500):",
          f"new cap = {cap} steps" if cap else "no cap below the trial length; discuss")


if __name__ == "__main__":
    main()
