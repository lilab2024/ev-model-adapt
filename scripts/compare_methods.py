"""Statistical comparison of the reported methods (D035).

Usage (from the repository root):
    python scripts/compare_methods.py

Reads results/<method>/runs.csv of the methods in config/statistics.yaml and writes to
results/statistics/: vehicle_values.csv, intervals.csv (confirmatory Q1 pairs at L2 with
Holm-corrected bootstrap p-values; all other pairs with intervals only), vehicle_differences.csv
(vehicle-by-vehicle differences, for every level), collisions.csv and run_info.json, which
records the git commit of every results folder (they should all come from one final re-run).
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ema import config, data, protocol
from ema import comparison as st


def git_state():
    root = config.REPO_ROOT
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=root).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                           cwd=root).stdout.strip() != ""
    return head, dirty


def main():
    cfg = config.load_yaml("statistics.yaml")
    cfg_e = config.load_yaml("evaluation.yaml")
    head, dirty = git_state()
    res = config.REPO_ROOT / "results"
    out = res / "statistics"
    out.mkdir(parents=True, exist_ok=True)
    lengths = [float(L) for L in cfg["lengths"]]

    runs, provenance = {}, {}
    for short, folder in cfg["methods"].items():
        path = res / folder / "runs.csv"
        if not path.exists():
            print(f"missing, skipped: {folder}", flush=True)
            continue
        runs[short] = pd.read_csv(path)
        info = json.loads((res / folder / "run_info.json").read_text(encoding="utf-8"))
        provenance[short] = dict(folder=folder, git_commit=info.get("git_commit", "?"),
                                 git_dirty=info.get("git_dirty"))
    values = {k: st.vehicle_values(v) for k, v in runs.items()}

    conf_cfg = cfg["confirmatory"]
    conf_pairs = [tuple(p) for p in conf_cfg["pairs"]]
    conf = st.compare(values, conf_pairs, conf_cfg["level"], lengths, cfg, "confirmatory")
    parts = [conf]
    for level in cfg["exploratory"]["levels"]:
        pairs = [tuple(p) for p in cfg["exploratory"]["pairs"]
                 if not (level == conf_cfg["level"] and tuple(p) in conf_pairs)]
        parts.append(st.compare(values, pairs, level, lengths, cfg, "exploratory"))
    intervals = pd.concat([p for p in parts if len(p)], ignore_index=True)
    all_pairs = list(dict.fromkeys(conf_pairs + [tuple(p) for p in cfg["exploratory"]["pairs"]]))
    desc = st.describe(values, all_pairs, cfg["descriptive_levels"], lengths)
    rt = protocol.run_table(data.load_runs(), cfg_e)
    coll = st.collisions(runs, rt["is_stop_run"])

    pd.concat([v.assign(method=k) for k, v in values.items()], ignore_index=True).to_csv(
        out / "vehicle_values.csv", index=False)
    intervals.to_csv(out / "intervals.csv", index=False)
    desc.to_csv(out / "vehicle_differences.csv", index=False)
    coll.to_csv(out / "collisions.csv", index=False)
    commits = sorted({p["git_commit"] for p in provenance.values()})
    (out / "run_info.json").write_text(json.dumps(dict(
        git_commit=head, git_dirty=dirty,
        config_md5=hashlib.md5((config.CONFIG_DIR / "statistics.yaml").read_bytes()).hexdigest(),
        results=provenance, n_result_commits=len(commits)), indent=2), encoding="utf-8")

    print(f"Saved to {out} (git {head[:7]}{' DIRTY' if dirty else ''})")
    if len(commits) > 1:
        print(f"WARNING: results come from {len(commits)} commits: {[c[:7] for c in commits]}")
    show = intervals.copy()
    for c in ["mean_diff", "ci_lo", "ci_hi"]:
        show[c] = show[c].round(2)
    for c in ["p", "p_holm"]:
        show[c] = show[c].round(4)
    with pd.option_context("display.width", 220, "display.max_rows", 300):
        print(show.to_string(index=False))
    print("collided runs:", int(coll["n_collided"].sum()), "of", int(coll["n_runs"].sum()))


if __name__ == "__main__":
    main()
