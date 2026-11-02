"""Figures and tables of the paper (D034, D037, D040, D041, D043, D044).

Usage (from the repository root):
    python scripts/make_figures.py                     # final: results/figures, results/tables
    python scripts/make_figures.py --draft DIR         # style review: DIR/figures, DIR/tables
    python scripts/make_figures.py --draft DIR --only fig_q1_l2 tab_q1

Reads only finished results (results/statistics, results/<method>/fold_states.csv and
runs.csv, results/diagnostics, results/m4_pretrain_*) and the run table of the data. A final
run refuses uncommitted code (D044 (2)); a draft run may use it and says so in run_info.json.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ema import config  # noqa: E402
from ema import figures as F  # noqa: E402

GROUP1 = ["tab_data", "tab_q1", "fig_q1_l2", "fig_ai_methods", "fig_split_prior", "fig_split_validation",
          "fig_ablation_pretraining", "fig_method_vehicle", "fig_selected_steps", "fig_idm_params",
          "tab_idm_params", "fig_physics_floors", "tab_physics_floors", "fig_l3_vehicles",
          "tab_pretrain_diagnostics", "tab_compute", "tab_collisions", "tab_all_methods"]
GROUP2 = ["tab_gbm_importance", "fig_collision_example", "fig_residual_map"]   # recomputed, checked (D050, D052)
GROUP3 = ["fig_traj_example", "fig_levels_schematic", "fig_methods_schematic"]   # D043, D055, D056
ALL = GROUP1 + GROUP2 + GROUP3


def git_state(root):
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=root).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                           cwd=root).stdout.strip() != ""
    return head, dirty


def md5_of(paths):
    h = hashlib.md5()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", type=Path, default=None, help="output folder for a style review")
    ap.add_argument("--only", nargs="*", default=None, help="names of the outputs to make")
    ap.add_argument("--results", type=Path, default=None, help="results folder (tests only)")
    ap.add_argument("--reuse-compute", action="store_true",
                    help="draft only: reuse the group-2 computations of an earlier draft with the same code")
    args = ap.parse_args(argv)

    root = config.REPO_ROOT
    res = args.results if args.results is not None else root / "results"
    head, dirty = git_state(root)
    if args.draft is None and (dirty or args.results is not None):
        raise SystemExit("final run refused: uncommitted changes (D044); use --draft for a review")
    base = args.draft if args.draft is not None else res
    fig_dir, tab_dir = base / "figures", base / "tables"
    wanted = args.only or ALL
    unknown = sorted(set(wanted) - set(ALL))
    if args.reuse_compute and args.draft is None:
        raise SystemExit("--reuse-compute is for drafts only; a final run recomputes everything")
    if unknown:
        raise SystemExit(f"unknown outputs: {unknown}")

    cfg = config.load_yaml("figures.yaml")
    cfg_stat = config.load_yaml("statistics.yaml")
    cfg_models = config.load_yaml("models.yaml")
    bounds = {k: v for k, v in cfg_models["idm"]["params"].items() if isinstance(v, list)}
    lengths = [float(L) for L in cfg_stat["lengths"]]
    plt = F.setup(cfg)
    folder = cfg_stat["methods"]                                  # short name -> results folder

    st_dir = res / "statistics"
    st_info = json.loads((st_dir / "run_info.json").read_text(encoding="utf-8"))
    vv = pd.read_csv(st_dir / "vehicle_values.csv")
    values = {m: g.drop(columns="method") for m, g in vv.groupby("method")}
    intervals = pd.read_csv(st_dir / "intervals.csv")
    coll = pd.read_csv(st_dir / "collisions.csv")
    states = {m: pd.read_csv(res / f / "fold_states.csv") for m, f in folder.items()
              if (res / f / "fold_states.csv").exists()}
    made = []

    def fig_out(name, fig):
        F.save(fig, fig_dir, name, cfg["style"]["dpi_png"])
        plt.close(fig)
        made.append(name)
        print(f"figure {name}", flush=True)

    def tab_out(name, df, label, **kw):
        extra = cfg.get("table_notes", {}).get(name)              # D058: reader-facing explanations
        if extra:
            kw["note"] = kw["note"] + " " + extra if kw.get("note") else extra
        F.save_table(df, tab_dir, name, cfg["captions"][name], label, **kw)
        made.append(name)
        print(f"table  {name} ({len(df)} rows)", flush=True)

    g = cfg["groups"]
    if "tab_data" in wanted:
        from ema import data, protocol
        rt = protocol.run_table(data.load_runs(), config.load_yaml("evaluation.yaml"))
        nl = chr(10)
        tab_out("tab_data", F.data_overview(rt), "tab:data", float_fmt="{:.1f}",
                header=["Powertrain", "Vehicle", "Gap" + nl + "settings", "Runs", "Configu-" + nl + "rations",
                        "Hours", "Stop" + nl + "runs"])
    if "tab_q1" in wanted:
        t, latex = F.confirmatory_table(intervals, cfg)
        tab_out("tab_q1", t, "tab:q1", **latex)
    if "fig_q1_l2" in wanted:
        fig_out("fig_q1_l2", F.curves_figure(plt, values, g["q1"], "L2_new_vehicle", lengths, cfg, cfg_stat,
                                             "q1_l2", log_y=True))
    if "fig_ai_methods" in wanted:
        fig_out("fig_ai_methods", F.curves_figure(plt, values, g["ai"], "L2_new_vehicle", lengths, cfg,
                                                  cfg_stat, "ai_methods", reference=g["ai_reference"],
                                                  intervals=False))
    for name, (a, b) in cfg["splits"].items():
        if name in wanted:
            d = F.split_differences(values, a, b, "L2_new_vehicle", lengths)
            la, lb = cfg["methods"][a]["label"], cfg["methods"][b]["label"]
            title = f"{la} {chr(8722)} {lb} (negative: first better)"
            fig_out(name, F.split_figure(plt, d, cfg, title, lengths))
    if "fig_method_vehicle" in wanted:
        fig_out("fig_method_vehicle", F.vehicle_heatmap(plt, values, g["heatmap"], "L2_new_vehicle",
                                                        cfg["example_length_s"], cfg))
    if "fig_selected_steps" in wanted:
        runs = pd.read_csv(res / folder["M4A_lovo"] / "runs.csv", usecols=["fold_id", "task"])
        steps = F.selected_steps(states["M4A_ft"], states["M4A_lovo"], F.fold_test_info(runs))
        fig_out("fig_selected_steps", F.selected_steps_figure(plt, steps, cfg))
    if "fig_idm_params" in wanted or "tab_idm_params" in wanted:
        params = F.idm_task_params(pd.read_csv(res / "diagnostics" / "idm_floor" / "idm_floor_runs.csv"), cfg)
        if "fig_idm_params" in wanted:
            fig_out("fig_idm_params", F.idm_params_figure(plt, params, cfg, bounds))
        if "tab_idm_params" in wanted:
            t, latex = F.idm_params_table(params, bounds)
            tab_out("tab_idm_params", t, "tab:idm_params", **latex)
    if "fig_physics_floors" in wanted:
        tasks = pd.read_csv(res / "diagnostics" / "physics_floor" / "physics_floor_tasks.csv")
        fig_out("fig_physics_floors", F.physics_floors_figure(plt, tasks, cfg))
    if "tab_physics_floors" in wanted:
        summ = pd.read_csv(res / "diagnostics" / "physics_floor" / "physics_floor_summary.csv")
        t, latex = F.physics_floors_table(summ, cfg)
        tab_out("tab_physics_floors", t, "tab:physics_floors", **latex)
    if "fig_l3_vehicles" in wanted:
        fig_out("fig_l3_vehicles", F.l3_vehicles_figure(plt, values, g["l3"], cfg))
    if "tab_pretrain_diagnostics" in wanted:
        diags, notes = {}, []
        for label, p in cfg["pretrained"].items():
            diags[label] = pd.read_csv(res / p["dir"] / "diagnostics.csv")
            info = json.loads((res / p["dir"] / "run_info.json").read_text(encoding="utf-8"))
            notes.append(f"{label}: {info.get('steps_run')} training steps, best validation loss "
                         f"{info.get('best_val_loss'):.2f}")          # D058: no weights md5 or git commit
        t, latex = F.pretrain_diagnostics_table(diags)
        n_ep = sorted(set(t["episodes"]))
        tab_out("tab_pretrain_diagnostics", t, "tab:pretrain",
                note=f"Each value: {', '.join(map(str, n_ep))} synthetic validation episodes. " + "; ".join(notes) + ".",
                **latex)
    if "tab_compute" in wanted:
        t, devices, latex, folds = F.compute_table({m: states[m] for m in g["compute"]}, cfg)
        tab_out("tab_compute", t, "tab:compute", float_fmt="{:.0f}",
                note=folds + f" GPU: {', '.join(devices)}.", **latex)
    if "tab_collisions" in wanted:
        t, others, stop_veh, latex = F.collisions_table(coll, cfg)
        note = (f"All other methods: {others} collided simulations. Stop runs (EV runs braking to "
                f"standstill) exist only for: {', '.join(stop_veh)}.")
        tab_out("tab_collisions", t, "tab:collisions", note=note, **latex)
    if "tab_all_methods" in wanted:
        t, latex = F.all_methods_table(values, coll, cfg, g["all"], cfg["table_levels"], cfg["table_lengths"])
        tab_out("tab_all_methods", t, "tab:all_methods", **latex)

    selections = {}
    if "fig_traj_example" in wanted or "fig_levels_schematic" in wanted:
        from ema import data, protocol
        cfg_eval = config.load_yaml("evaluation.yaml")
        rt_ex = protocol.run_table(data.load_runs(), cfg_eval)
    if "fig_traj_example" in wanted:
        ex = F.pick_traj_example(rt_ex, data.load_trajectories(columns=["t", "v_l", "s"]), cfg)
        selections["traj_example"] = ex["info"]
        print(f"traj example: {ex['info']}", flush=True)
        fig_out("fig_traj_example", F.traj_example_figure(plt, ex, cfg))
    if "fig_levels_schematic" in wanted:
        folds = protocol.make_folds(rt_ex, cfg_eval)
        selections["levels_schematic"] = {lv: next(f["fold_id"] for f in folds if f["level"] == lv)
                                          for lv in cfg["schematics"]["level_order"]}
        fig_out("fig_levels_schematic", F.levels_schematic_figure(plt, folds, F.schematic_tasks(rt_ex), cfg))
    if "fig_methods_schematic" in wanted:
        fig_out("fig_methods_schematic", F.methods_schematic_figure(plt, cfg))

    checks = {}
    if any(n in wanted for n in GROUP2):
        checks = group2(wanted, base, res, cfg, plt, fig_out, tab_out, head, args.reuse_compute)

    info = dict(git_commit=head, git_dirty=dirty, draft=args.draft is not None, checks=checks, selections=selections,
                config_md5=md5_of([config.CONFIG_DIR / "figures.yaml", config.CONFIG_DIR / "statistics.yaml"]),
                statistics_git_commit=st_info.get("git_commit"),
                statistics_result_commits=st_info.get("n_result_commits"), outputs=made)
    base.mkdir(parents=True, exist_ok=True)
    (base / "figures_run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"{len(made)} outputs in {base} (git {head[:7]}{' DIRTY' if dirty else ''}"
          f"{', draft' if args.draft is not None else ''})")


def group2(wanted, base, res, cfg, plt, fig_out, tab_out, head, reuse):
    """The outputs that recompute and check (D050, D052); returns the reproduction checks."""
    import pickle

    from ema import figures_compute as FC
    cfg_e = config.load_yaml("evaluation.yaml")
    cfg_m = config.load_yaml("models.yaml")
    key = md5_of([Path(FC.__file__), config.CONFIG_DIR / "figures.yaml"]) + head
    cache_dir = base / "compute_cache"
    prot = None

    def compute(name, fn):
        nonlocal prot
        path = cache_dir / f"{name}.pkl"
        if reuse and path.exists():
            with open(path, "rb") as f:
                saved = pickle.load(f)
            if saved["key"] == key:
                print(f"{name}: reusing the draft computation", flush=True)
                return saved["value"]
        if prot is None:
            print("loading data and protocol ...", flush=True)
            prot = FC.load_protocol(cfg_e)
        value = fn(prot)
        if base != res:                                   # drafts only: keep for --reuse-compute
            cache_dir.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(dict(key=key, value=value), f)
        return value

    checks = {}
    if "tab_gbm_importance" in wanted:
        print("tab_gbm_importance: refitting features_gbm on the L2 folds ...", flush=True)
        imp, checks["gbm"] = compute("gbm", lambda p: FC.gbm_importance(
            p, cfg_m, cfg_e, res, use_floor_targets=cfg["gbm"]["use_floor_targets"], rtol=float(cfg["gbm"]["rtol"])))
        t, latex = F.gbm_importance_table(imp, cfg, n_folds=len(checks["gbm"]))
        tab_out("tab_gbm_importance", t, "tab:gbm_importance", **latex)
    if "fig_collision_example" in wanted:
        ex, checks["collision"] = compute("collision", lambda p: FC.collision_example(p, cfg_m, cfg_e, res))
        print(f"collision example: {checks['collision']}", flush=True)
        fig_out("fig_collision_example", F.collision_figure(plt, ex, cfg))
    if "fig_residual_map" in wanted:
        rm, checks["residual"] = compute("residual", lambda p: FC.residual_map(p, cfg_m, cfg_e, res, cfg))
        print(f"residual map: {rm['n_folds']} folds, re-simulation identical: "
              f"{all(c['identical'] for c in checks['residual'])}", flush=True)
        fig_out("fig_residual_map", F.residual_map_figure(plt, rm, cfg))
    return checks


if __name__ == "__main__":
    main()
