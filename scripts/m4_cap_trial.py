"""Validation-only trials for the fine-tuning settings of m4a_finetuned (F026, F028).

Usage (from the repository root):
    python scripts/m4_cap_trial.py                       # section cap_trial (F026)
    python scripts/m4_cap_trial.py --section lr_trial    # learning-rate trial (F028)

For each fold in the section's folds (config/m4.yaml), the pretrained network is
fine-tuned as in m4a_finetuned, but up to the section's max_steps, with its lr if given
and, with early_stopping false, without early stopping. Only the validation curve is kept,
in results/m4_<section>/<fold>/ (history.csv, run_info.json); no test run is simulated and
no network is saved. A fold is skipped if its saved run_info.json comes from the same clean
commit and config files. A section with a reference is compared with that earlier trial
by the rule fixed before the trial (cap_trial.lr_decision); otherwise the cap rule
(cap_trial.choose_cap) is applied.
"""
import argparse
import datetime
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ema import calibrate, cap_trial, config, data, protocol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", default="cap_trial", choices=["cap_trial", "lr_trial"])
    section = ap.parse_args().section
    cfg_e, cfg_m = config.load_yaml("evaluation.yaml"), config.load_yaml("models.yaml")
    cfg_pre, cfg_m4 = config.load_yaml("pretraining.yaml"), config.load_yaml("m4.yaml")
    trial = cap_trial.trial_config(cfg_m4, section)
    ft, ct = trial["finetuning"], cfg_m4[section]
    head, dirty = cap_trial.git_state()
    cfg_hash = cap_trial.config_md5()
    print(f"{section}: folds {ct['folds']}, lr {ft['lr']}, max steps {ft['max_steps']}, patience "
          f"{ft['patience']} (early stopping {'on' if ct.get('early_stopping', True) else 'off'}), "
          f"git {head[:7]}{' DIRTY' if dirty else ''}", flush=True)

    references = None
    if "reference" in ct:                               # check the reference before training
        ref_ct = cfg_m4[ct["reference"]]
        ref_dir = config.REPO_ROOT / "results" / f"m4_{ct['reference']}"
        references = [cap_trial.check_reference(ref_dir / fid, ct["reference_commit"],
                                                cfg_m4["finetuning"]["lr"], ref_ct["max_steps"])
                      for fid in ct["folds"]]
        print(f"reference: {ref_dir} (git {ct['reference_commit']}, checked)", flush=True)

    rt = protocol.run_table(data.load_runs(), cfg_e)
    folds = [f for f in protocol.make_folds(rt, cfg_e) if f["fold_id"] in ct["folds"]]
    missing = set(ct["folds"]) - {f["fold_id"] for f in folds}
    if missing:
        raise ValueError(f"unknown folds in {section}.folds: {sorted(missing)}")
    folds = sorted(folds, key=lambda f: ct["folds"].index(f["fold_id"]))
    assignment = protocol.assign_runs(rt, folds, cfg_e)
    traj_fd = calibrate.forward_difference(data.load_trajectories(), cfg_e["simulation"]["dt_s"])
    out = config.REPO_ROOT / "results" / f"m4_{section}"

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
            trial_section=section, trial=ct, finetuning=ft, n_train_runs=len(train_ids), **info,
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
    if references is None:
        cap = cap_trial.choose_cap(histories, ft["max_steps"])
        print("rule (D022, tol 1 %, grid 500):",
              f"new cap = {cap} steps" if cap else "no cap below the trial length; discuss")
        return
    tol = float(ct.get("tol", 0.01))
    adopt, cap, rows = cap_trial.lr_decision(histories, references, ft["max_steps"], tol)
    for f, r in zip(folds, rows):
        print(f"{f['fold_id']}: best {r['best_new']:.3f} vs reference {r['best_ref']:.3f} "
              f"(limit {r['limit']:.3f}); unstable: {r['unstable'] or 'no'}; "
              f"{'pass' if r['ok'] else 'FAIL'}")
    if not adopt:
        print(f"rule (F028): keep lr {cfg_m4['finetuning']['lr']}; cap "
              f"{cfg_m4[ct['reference']]['max_steps']} steps with early stopping (option 1)")
    else:
        print(f"rule (F028): adopt lr {ft['lr']};",
              f"new cap = {cap} steps" if cap else "no cap below the trial length; discuss")


if __name__ == "__main__":
    main()
