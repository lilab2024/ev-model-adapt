"""Figures and tables (D037, D040, D041, D043, D044): helpers, one-axes final-size figures,
tables and an end-to-end draft run of scripts/make_figures.py on synthetic results."""
import json
import sys

import numpy as np
import pandas as pd
import pytest

from ema import config, figures as F

CFG = config.load_yaml("figures.yaml")
CFG_STAT = dict(config.load_yaml("statistics.yaml"), n_boot=200)
BOUNDS = {k: v for k, v in config.load_yaml("models.yaml")["idm"]["params"].items() if isinstance(v, list)}
LENGTHS = [10.0, 30.0, 60.0, 120.0, 300.0]
EVS = ["IONIQ 5", "Model 3", "Polestar 2"]
ICES = [f"Veh{c}" for c in "ABCDEF"]
LEVEL_VEH = {"L1_new_setting": [("EV", "IONIQ 5")] + [("ICE", v) for v in ICES],
             "L2_new_vehicle": [("EV", v) for v in EVS] + [("ICE", v) for v in ICES],
             "L2_no_ice": [("EV", v) for v in EVS], "L3_cross_powertrain": [("EV", v) for v in EVS]}
sys.path.insert(0, str(config.REPO_ROOT / "scripts"))


@pytest.fixture(scope="module")
def plt():
    return F.setup(CFG)


def _png_size(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def _check_fig(fig, width_key, height_key, tmp_path, name):
    assert len(fig.axes) == 1                                        # D037 (1)
    assert fig.axes[0].get_title() == ""                             # D037 (3)
    assert fig.axes[0].get_legend() is None                          # D046: legends below the axes
    if name not in ("heat", "methods"):
        assert len(fig.legends) == 1
        fig.canvas.draw()
        leg = fig.legends[0].get_window_extent()
        assert leg.y1 <= fig.axes[0].get_window_extent().y0 + 1       # below the axes, inside the figure
        assert leg.y0 >= 0
    if fig.axes[0].get_xscale() == "log" and name in ("q1", "ai", "split"):
        assert fig.axes[0].get_xlabel() == CFG["x_label"]
    w, h = CFG["style"][width_key], CFG["heights_in"][height_key]
    assert fig.get_size_inches() == pytest.approx([w, h])
    F.save(fig, tmp_path, name)
    assert _png_size(tmp_path / f"{name}.png") == (round(w * 300), round(h * 300))   # D037 (2)
    assert (tmp_path / f"{name}.pdf").exists()


# ---------------------------------------------------------------- synthetic results

def _vehicle_values(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for m in CFG["groups"]["all"]:
        levels = ["L1_new_setting", "L2_new_vehicle"] if m in ("M2b", "M2b_residual") else list(LEVEL_VEH)
        for lvl in levels:
            for pt, veh in LEVEL_VEH[lvl]:
                for L in ([np.nan] if m == "M0" else LENGTHS):
                    base = 10.0 if m == "M0" else 6.0 - (0 if np.isnan(L) else np.log10(L))
                    rows.append(dict(level=lvl, powertrain=pt, vehicle=veh, length_s=L,
                                     value=base + rng.uniform(0, 1), method=m))
    return pd.DataFrame(rows)


def _intervals():
    rows = []
    for a, b in config.load_yaml("statistics.yaml")["confirmatory"]["pairs"]:
        for L in LENGTHS:
            rows.append(dict(kind="confirmatory", level="L2_new_vehicle", method=a, versus=b, length_s=L,
                             n_vehicles=9, mean_diff=-1.0, ci_lo=-2.0, ci_hi=-0.1, n_better=7, p=0.002,
                             p_holm=0.012))
    return pd.DataFrame(rows)


def _collisions():
    rows = []
    for m in CFG["groups"]["all"]:
        for lvl, vehs in LEVEL_VEH.items():
            for pt, veh in vehs:
                for rt in ("stop", "no_stop"):
                    n = 0 if (rt == "stop" and veh != "IONIQ 5") else 100
                    c = 3 if (m == "M4C" and lvl == "L3_cross_powertrain" and rt == "stop" and n) else 0
                    rows.append(dict(method=m, level=lvl, vehicle=veh, run_type=rt, n_runs=n, n_collided=c,
                                     share=c / n if n else np.nan))
    return pd.DataFrame(rows)


def _lovo_folds():
    fids = ([f"L2_new_vehicle_0{i}" for i in range(1, 10)] + [f"L2_no_ice_0{i}" for i in range(1, 4)]
            + ["L3_cross_powertrain_01"])
    tasks = ([[f"EV | {v} | Short"] for v in EVS] + [[f"ICE | {v} | min", f"ICE | {v} | max"] for v in ICES]
             + [[f"EV | {v} | Short"] for v in EVS] + [[f"EV | {v} | Short" for v in EVS]])
    return fids, tasks


def _states(name, fids, rng):
    lvl = [f.rsplit("_", 1)[0] for f in fids]
    if name == "M2b_residual":
        return pd.DataFrame(dict(fold_id=fids, level=lvl, res_train_s=rng.uniform(600, 9000, len(fids)),
                                 res_best_step=rng.integers(0, 30, len(fids)) * 100, res_gpu="RTX"))
    return pd.DataFrame(dict(fold_id=fids, level=lvl, best_step=rng.integers(1, 60, len(fids)) * 50,
                             finetune_s=rng.uniform(600, 9000, len(fids)), gpu_name="RTX"))


def make_results(root):
    """A small results tree with the files read by scripts/make_figures.py."""
    rng = np.random.default_rng(3)
    st = root / "statistics"
    st.mkdir(parents=True)
    _vehicle_values().to_csv(st / "vehicle_values.csv", index=False)
    _intervals().to_csv(st / "intervals.csv", index=False)
    _collisions().to_csv(st / "collisions.csv", index=False)
    (st / "run_info.json").write_text(json.dumps(dict(git_commit="c3e6d75", n_result_commits=1)))
    folder = config.load_yaml("statistics.yaml")["methods"]
    fids, tasks = _lovo_folds()
    ft_fids = [f"L1_new_setting_{i:02d}" for i in range(1, 17)] + fids
    for m in CFG["groups"]["compute"]:
        d = root / folder[m]
        d.mkdir(parents=True)
        f = ft_fids if m == "M4A_ft" else ([x for x in ft_fids if not x.startswith(("L2_no", "L3"))]
                                          if m == "M2b_residual" else fids)
        _states(m, f, rng).to_csv(d / "fold_states.csv", index=False)
    pd.DataFrame([dict(fold_id=f, task=t) for f, ts in zip(fids, tasks) for t in ts]).to_csv(
        root / folder["M4A_lovo"] / "runs.csv", index=False)
    idm = root / "diagnostics" / "idm_floor"
    idm.mkdir(parents=True)
    rows = []
    for k, (pt, veh) in enumerate(LEVEL_VEH["L2_new_vehicle"]):
        for gap in (["Short", "Long"] if pt == "EV" else ["min", "max"]):
            for floor in ("in_sample", "cross_config"):
                rows.append(dict(task=f"{pt} | {veh} | {gap}", run_id=f"r{k}{gap}", floor=floor,
                                 idm_v0=30.0 + k, idm_T=1.0 + 0.1 * k + (0.3 if gap in ("Long", "max") else 0),
                                 idm_s0=2.0 + 0.1 * k, idm_a=1.0, idm_b=1.5, powertrain=pt))
    pd.DataFrame(rows).to_csv(idm / "idm_floor_runs.csv", index=False)
    phy = root / "diagnostics" / "physics_floor"
    phy.mkdir(parents=True)
    tr, sm = [], []
    for model in ("idm", "ovrv", "idm_lag"):
        for pt, vehs in (("EV", EVS), ("ICE", ICES)):
            for floor in ("in_sample", "cross_config"):
                vals = rng.uniform(0.8, 4.0, len(vehs))
                for v, x in zip(vehs, vals):
                    tr.append(dict(powertrain=pt, task=f"{pt} | {v} | Short", model=model, floor=floor, n_runs=4,
                                   spacing_rmse_mean=x, n_collided=0))
                sm.append(dict(model=model, powertrain=pt, floor=floor, n_tasks=len(vehs),
                               task_mean_rmse=vals.mean(), task_median_rmse=np.median(vals), n_collided=0))
    pd.DataFrame(tr).to_csv(phy / "physics_floor_tasks.csv", index=False)
    pd.DataFrame(sm).to_csv(phy / "physics_floor_summary.csv", index=False)
    for label, p in CFG["pretrained"].items():
        d = root / p["dir"]
        d.mkdir(parents=True)
        pd.DataFrame(dict(length_s=[10.0, 30.0, 60.0], n=392, rmse_pred=[6.0, 5.0, 4.0], rmse_true=0.08,
                          rmse_pop=13.0)).to_csv(d / "diagnostics.csv", index=False)
        (d / "run_info.json").write_text(json.dumps(dict(steps_run=8900, best_val_loss=5.3, git_commit="0529b33")))
    return root


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    return make_results(tmp_path_factory.mktemp("results"))


def _values():
    vv = _vehicle_values()
    return {m: g.drop(columns="method") for m, g in vv.groupby("method")}


# ---------------------------------------------------------------- helpers and tables

def test_parse_task_gap_rank_spread_labels():
    assert F.parse_task("EV | IONIQ 5 | Short") == ("EV", "IONIQ 5", "Short")
    assert F.gap_rank("min", CFG) < F.gap_rank("max", CFG)
    assert F.gap_rank("Short", CFG) < F.gap_rank("XLong", CFG)
    y = np.array([1.0, 1.01, 0.99, 5.0, 1.02])
    out = F.spread_labels(y, 0.1)
    assert np.all(np.diff(out[np.argsort(y, kind="stable")]) >= 0.1 - 1e-12)


def test_latex_table_bold_note_wide_groups():
    df = pd.DataFrame({"a_b": ["x%", "y"], "v": [1.234, -2.5], "p (Holm)": [0.01234, 0.5], "r": ["17-99", "a_b"]})
    bold = pd.DataFrame([[False, True, False, False], [False] * 4], columns=df.columns)
    s = F.latex_table(df, "cap", "tab:x", bold=bold, note="n", wide=True, row_groups=[("G1", 1), ("G2", 1)],
                      raw_cols=("r",), size="footnotesize")
    bs = chr(92)
    assert s.startswith(bs + "begin{table*}") and s.rstrip().endswith(bs + "end{table*}")
    assert bs + "textbf{1.23}" in s and "0.0123" in s and "x" + bs + "%" in s
    assert "$-$2.50" in s and "17-99" in s and "a_b" in s                        # raw column untouched
    assert s.count(bs + "textit{") == 2 and bs + "addlinespace" in s
    assert F._latex_cell("17-99", "{:.2f}") == "17--99" and F._latex_cell(-0.001, "{:.2f}") == "0.00"


def test_idm_params_and_table(results):
    p = F.idm_task_params(pd.read_csv(results / "diagnostics/idm_floor/idm_floor_runs.csv"), CFG)
    assert len(p) == 18
    assert list(p.loc[p["vehicle"] == "VehA", "gap"]) == ["min", "max"]
    t, latex = F.idm_params_table(p, BOUNDS)
    assert list(t.columns[:3]) == ["Powertrain", "Vehicle", "Gap"] and len(t) == 18
    tex = F.latex_table(latex.pop("latex_df"), "c", "t", **latex)
    assert "$v_0$" + chr(92) + chr(92) + "(m/s)" in tex and "Bounds of the fit" in tex and "IONIQ 5 (EV)" in tex
    bad = pd.read_csv(results / "diagnostics/idm_floor/idm_floor_runs.csv")
    bad = pd.concat([bad, bad.iloc[[1]].assign(idm_T=9.0, run_id="extra")])
    with pytest.raises(ValueError):
        F.idm_task_params(bad, CFG)


def test_fold_test_info_and_selected_steps(results):
    folder = config.load_yaml("statistics.yaml")["methods"]
    info = F.fold_test_info(pd.read_csv(results / folder["M4A_lovo"] / "runs.csv"))
    lab = info.set_index("fold_id")["test_label"]
    assert lab["L2_new_vehicle_01"] == "IONIQ 5" and lab["L3_cross_powertrain_01"] == "all EVs"
    ft = pd.read_csv(results / folder["M4A_ft"] / "fold_states.csv")
    lv = pd.read_csv(results / folder["M4A_lovo"] / "fold_states.csv")
    s = F.selected_steps(ft, lv, info)
    assert len(s) == 13 and s["step_original"].notna().all()
    with pytest.raises(ValueError):
        F.selected_steps(ft, lv, info.iloc[:3])


def test_q1_table(results):
    t, latex = F.confirmatory_table(_intervals(), CFG)
    assert len(t) == 20 and [n for _, n in latex["row_groups"]] == [5, 5, 5, 5]
    assert "comparison" not in latex["latex_df"].columns
    tex = F.latex_table(latex.pop("latex_df"), "c", "t", **latex)
    assert "Population prior vs No adaptation" in tex and "95" in tex


def test_tables_compute_collisions_all_methods_pretrain(results):
    folder = config.load_yaml("statistics.yaml")["methods"]
    states = {m: pd.read_csv(results / folder[m] / "fold_states.csv") for m in CFG["groups"]["compute"]}
    c, devices, _, folds = F.compute_table(states, CFG)
    assert folds.startswith("Folds: 29 ")
    assert len(c) == 6 and c["time/fold median (min)"].gt(0).all() and devices == ["RTX"]
    assert "device" not in c.columns
    t, others, stop_veh, _ = F.collisions_table(_collisions(), CFG)
    assert others == 0 and stop_veh == ["IONIQ 5"]
    assert t.loc[t["Level"] == "New EV, ICE-only training", "stop runs: collided/simulations"].item() == "3/100"
    a, latex = F.all_methods_table(_values(), _collisions(), CFG, CFG["groups"]["all"],
                                   CFG["table_levels"], CFG["table_lengths"])
    bold = latex["bold"]
    assert len(latex["header"]) == len(a.columns) and sum(n for _, n in latex["groups"]) == len(a.columns)
    tex = F.latex_table(a, "c", "t", **latex)
    assert chr(92) + "footnotesize" in tex
    assert "New EV," + chr(92) + chr(92) + "EV-only training" in tex and chr(92) + "cmidrule(lr){3-5}" in tex
    assert len(a) == 12 and a.columns[0] == "Method" and "L2 10 s" in a.columns
    assert a.loc[a["Method"] == CFG["methods"]["M2b"]["label"], "L3 10 s"].isna().all()
    assert bold["L2 10 s"].sum() >= 1 and not bold["Method"].any()
    assert a.loc[a["Method"] == CFG["methods"]["M4C"]["label"], "collisions"].item() == "3"
    d = {k: pd.read_csv(results / p["dir"] / "diagnostics.csv") for k, p in CFG["pretrained"].items()}
    t, latex = F.pretrain_diagnostics_table(d)
    assert len(t) == 9 and [n for _, n in latex["row_groups"]] == [3, 3, 3]
    tex = F.latex_table(latex.pop("latex_df"), "c", "t", **latex)
    assert tex.count(chr(92) + "textit{") == 3 and "Network (GRU encoder)" in tex


def test_physics_floors_table_order(results):
    s = pd.read_csv(results / "diagnostics/physics_floor/physics_floor_summary.csv")
    t, _ = F.physics_floors_table(s, CFG)
    assert list(t["Model"]) == ["IDM", "IDM", "OVRV", "OVRV", "IDM with lag", "IDM with lag"]
    assert list(t["Powertrain"][:2]) == ["EV", "ICE"]


# ---------------------------------------------------------------- figures

def test_curves_split_heatmap_l3(plt, tmp_path):
    v = _values()
    fig = F.curves_figure(plt, v, CFG["groups"]["q1"], "L2_new_vehicle", LENGTHS, CFG, CFG_STAT, "q1_l2",
                          log_y=True)
    _check_fig(fig, "single_width_in", "q1_l2", tmp_path, "q1")
    assert fig.axes[0].get_yscale() == "log"
    zero = [c for c in fig.axes[0].containers if c.get_label() == CFG["methods"]["M4A_zero"]["label"]]
    assert zero and zero[0].lines[0].get_markerfacecolor() == "white"            # D045: open marker
    fig = F.curves_figure(plt, v, CFG["groups"]["ai"], "L2_new_vehicle", LENGTHS, CFG, CFG_STAT, "ai_methods",
                          reference="M2b", intervals=False)
    assert not any(len(c) for c in fig.axes[0].containers if hasattr(c, "has_yerr") and c.has_yerr)
    _check_fig(fig, "single_width_in", "ai_methods", tmp_path, "ai")
    assert any("reference" in t.get_text() for t in fig.legends[0].get_texts())
    d = F.split_differences(v, "M2b", "M2", "L2_new_vehicle", LENGTHS)
    fig = F.split_figure(plt, d, CFG, "Powertrain prior - Population prior", LENGTHS)
    _check_fig(fig, "single_width_in", "split", tmp_path, "split")
    leg = fig.legends[0]
    assert [t.get_text() for t in leg.get_texts()] == [f"{v} (EV)" for v in EVS] + [f"{v} (ICE)" for v in ICES]
    assert leg.get_title().get_text().startswith("Powertrain prior") and not fig.axes[0].texts
    fig = F.vehicle_heatmap(plt, v, CFG["groups"]["heatmap"], "L2_new_vehicle", 60, CFG)
    _check_fig(fig, "double_width_in", "method_vehicle", tmp_path, "heat")
    assert len(fig.axes[0].texts) == 12 * 9
    assert {t.get_color() for t in fig.axes[0].texts} <= {"w", "k"}
    fig = F.l3_vehicles_figure(plt, v, CFG["groups"]["l3"], CFG)
    _check_fig(fig, "single_width_in", "l3_vehicles", tmp_path, "l3")
    assert len(fig.axes[0].get_yticklabels()) == 10
    assert len(fig.axes[0].lines) == 10 * 3 * 2 - 3                    # D048: M0 once per vehicle


def test_idm_steps_physics_figures(plt, tmp_path, results):
    p = F.idm_task_params(pd.read_csv(results / "diagnostics/idm_floor/idm_floor_runs.csv"), CFG)
    fig = F.idm_params_figure(plt, p, CFG, BOUNDS)
    _check_fig(fig, "single_width_in", "idm_params", tmp_path, "idm")
    ax = fig.axes[0]                                                  # D048: T per task, one row per vehicle
    assert [t.get_text() for t in ax.get_yticklabels()] == (ICES[::-1] + EVS[::-1])
    assert len(ax.texts) == len(p) and ax.get_xlabel().startswith("desired time gap")
    folder = config.load_yaml("statistics.yaml")["methods"]
    info = F.fold_test_info(pd.read_csv(results / folder["M4A_lovo"] / "runs.csv"))
    s = F.selected_steps(pd.read_csv(results / folder["M4A_ft"] / "fold_states.csv"),
                         pd.read_csv(results / folder["M4A_lovo"] / "fold_states.csv"), info)
    fig = F.selected_steps_figure(plt, s, CFG)
    _check_fig(fig, "single_width_in", "selected_steps", tmp_path, "steps")
    assert len(fig.axes[0].texts) == 13
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert "EV test vehicle" in labels and "New EV, ICE-only training" in labels and not any("L2" in t for t in labels)
    s0 = s.copy()
    s0.loc[0, "step_lovo"] = 0
    with pytest.raises(ValueError):
        F.selected_steps_figure(plt, s0, CFG)
    t = pd.read_csv(results / "diagnostics/physics_floor/physics_floor_tasks.csv")
    fig = F.physics_floors_figure(plt, t, CFG)
    _check_fig(fig, "single_width_in", "physics_floors", tmp_path, "phys")
    assert [x.get_text() for x in fig.axes[0].get_xticklabels()] == ["IDM", "OVRV", "IDM with lag"]
    xs = fig.axes[0].lines[0].get_xdata()                             # D048: offsets alternate around the centre
    assert abs(np.mean(xs) - np.median(xs)) < 0.05 and not np.all(np.diff(xs) > 0)


# ---------------------------------------------------------------- script

def test_make_figures_draft_run(results, tmp_path):
    import make_figures
    only = [n for n in make_figures.GROUP1 if n != "tab_data"]          # tab_data needs the data
    out = tmp_path / "draft"
    make_figures.main(["--results", str(results), "--draft", str(out), "--only", *only])
    info = json.loads((out / "figures_run_info.json").read_text())
    assert info["draft"] is True and sorted(info["outputs"]) == sorted(only)
    assert info["statistics_git_commit"] == "c3e6d75" and info["statistics_result_commits"] == 1
    for n in only:
        if n.startswith("fig_"):
            assert (out / "figures" / f"{n}.pdf").exists() and (out / "figures" / f"{n}.png").exists()
        else:
            assert (out / "tables" / f"{n}.csv").exists() and (out / "tables" / f"{n}.tex").exists()
    tex = (out / "tables" / "tab_collisions.tex").read_text()
    assert "IONIQ 5" in tex and chr(13) not in tex
    with pytest.raises(SystemExit):                                     # a final run needs real results
        make_figures.main(["--results", str(results), "--only", "tab_q1"])
    with pytest.raises(SystemExit):
        make_figures.main(["--results", str(results), "--draft", str(out), "--only", "fig_unknown"])


# ---------------------------------------------------------------- group 2 outputs (D050, D052)

def test_gbm_importance_table(tmp_path):
    import make_figures
    from ema import context_features
    rng = np.random.default_rng(0)
    imp = pd.DataFrame(rng.dirichlet(np.ones(18), size=5).T, index=list(context_features.FEATURE_NAMES),
                       columns=["v0", "T", "s0", "a", "b"])
    imp.loc["is_ev", "s0"], imp.loc["tg_mean", "s0"] = 0.0951, 0.0949            # print as 0.10 and 0.09
    t, latex = F.gbm_importance_table(imp, CFG, n_folds=9)
    assert "tab_gbm_importance" in make_figures.GROUP2 and "fig_gbm_importance" not in make_figures.ALL
    assert len(t) == 18 and sorted(t["feature"]) == sorted(CFG["gbm_feature_labels"].values())
    assert [n for _, n in latex["row_groups"]] == [2, 4, 4, 2, 3, 2, 1]
    assert t["feature"].iloc[0] == CFG["gbm_feature_labels"]["gap_intercept"] and t["feature"].iloc[-1] == "EV indicator"
    ldf, bold = latex["latex_df"], latex["bold"]
    row = {f: i for i, f in enumerate(ldf["Feature"])}
    assert bold.loc[row["EV indicator"], "s0"] and not bold.loc[row["time gap, mean"], "s0"]   # as printed
    small = ldf[["v0", "T", "s0", "a", "b"]] < 0.095
    assert not bold["Feature"].any() and not (bold[["v0", "T", "s0", "a", "b"]] & small).to_numpy().any()
    assert t[["v0", "T", "s0", "a", "b"]].sum().to_numpy() == pytest.approx(imp.sum().to_numpy())   # all rows kept
    F.save_table(t, tmp_path, "tab_gbm_importance", CFG["captions"]["tab_gbm_importance"], "tab:gbm", **latex)
    tex = (tmp_path / "tab_gbm_importance.tex").read_text()
    bs = chr(92)
    assert tex.startswith(bs + "begin{table}[t]") and tex.count(bs + "textit{") == 7 and "$s_0$" in tex
    assert bs + "textbf{0.10}" in tex and "over the 9 folds" in tex and chr(13) not in tex
    assert len(pd.read_csv(tmp_path / "tab_gbm_importance.csv")) == 18
    bad = CFG | {"gbm_feature_blocks": CFG["gbm_feature_blocks"][:-1]}
    with pytest.raises(ValueError):
        F.gbm_importance_table(imp, bad, n_folds=9)


def test_group2_figures(plt, tmp_path):
    t = np.arange(0, 60, 0.1)
    s_obs = 20 + 5 * np.sin(t / 10)
    s_net = s_obs - t / 2
    k = int(np.argmax(s_net <= 0))
    ex = dict(t=t, s_obs=s_obs, s_net=np.where(np.arange(len(t)) <= k, s_net, np.nan), n_valid=k + 1,
              s_prior=s_obs + 1)
    fig = F.collision_figure(plt, ex, CFG)
    _check_fig(fig, "single_width_in", "collision_example", tmp_path, "coll")
    assert [x.get_text() for x in fig.legends[0].get_texts()] == ["observed", CFG["methods"]["M4C"]["label"],
                                                                  "collision", CFG["methods"]["M2"]["label"]]
    gaps, dvs = np.linspace(2, 60, 40), np.linspace(-4, 4, 30)
    G, D = np.meshgrid(gaps, dvs)
    rm = dict(gaps=gaps, dvs=dvs, R=0.2 * np.tanh(D) - 0.1 * np.tanh((G - 30) / 10), v_med=17.3,
              H=np.exp(-((G - 25) / 10) ** 2 - (D / 1.5) ** 2), level=0.3, n_folds=6, n_states=1000,
              hx=gaps, hy=dvs)
    fig = F.residual_map_figure(plt, rm, CFG)
    _check_fig(fig, "single_width_in", "residual_map", tmp_path, "resid")
    assert "17.3" in fig.legends[0].get_texts()[-1].get_text() and len(fig.legends[0].get_texts()) == 3


# ---------------------------------------------------------------- group 3 (D043, D055)

def _traj_example_data():
    mph, dt = 0.44704, 0.1
    rows, traj = [], []

    def add(rid, pt, veh, gap, prof, vf, vr, stop, t, v_l):
        rows.append(dict(run_id=rid, powertrain=pt, vehicle=veh, gap=gap, profile=prof, v_free_mph=vf,
                         v_reduced_mph=vr, is_stop_run=stop))
        traj.append(pd.DataFrame(dict(run_id=rid, t=t, v_l=v_l, s=20 + 0.5 * np.sin(t / 7) + 0.1 * len(rid))))

    t_ice = np.arange(0, 620, dt)
    v_ice = np.full(len(t_ice), 24.1)
    for t0, depth in ((100, 2.4), (300, 4.0), (450, 7.7)):              # deepest dip at about 455 s
        m = (t_ice >= t0) & (t_ice < t0 + 20)
        v_ice[m] -= depth * np.sin(np.pi * (t_ice[m] - t0) / 20) ** 2
    for veh, gap in (("VehB", "min"), ("VehA", "max"), ("VehA", "min")):
        add(f"ICE_{veh}_dip_{gap}", "ICE", veh, gap, "dip", np.nan, np.nan, False, t_ice,
            v_ice + (0.3 if gap == "max" else 0.0))
        add(f"ICE_{veh}_osc_{gap}", "ICE", veh, gap, "osc", np.nan, np.nan, False, t_ice, v_ice - 5)
    t_ev = np.arange(0, 50, dt)
    for gap in ("XLong", "Short", "Long", "Medium"):
        for vf, vr in ((35, 15), (55, 35), (55, 45), (60, 35), (55, 0)):
            v = np.where(t_ev < 10, vf * mph, np.maximum(vr * mph, vf * mph - 2.0 * (t_ev - 10)))
            v = np.where(t_ev > 30, np.minimum(vf * mph, vr * mph + 1.0 * (t_ev - 30)), v)
            for k in (2, 1):
                add(f"EV_IONIQ5_{gap}_{vf}to{vr}_r{k}", "EV", "IONIQ 5", gap, "dip", vf, vr, vr == 0, t_ev, v)
    add("EV_Model3_Short_55to35_r1", "EV", "Model 3", "Short", "dip", 55, 35, False, t_ev, t_ev * 0 + 20)
    return pd.DataFrame(rows).set_index("run_id"), pd.concat(traj, ignore_index=True)


def test_leader_braking_onset():
    t = np.arange(0, 60, 0.1)
    v = np.where(t < 20, 24.0, np.maximum(16.0, 24.0 - 1.0 * (t - 20)))
    v = np.where(t > 40, np.minimum(24.0, 16.0 + 0.5 * (t - 40)), v)
    on, i_min = F.leader_braking_onset(t, v, 30.0, 0.5)
    assert t[i_min] == pytest.approx(28.0) and t[on] == pytest.approx(20.5)        # 24 - 0.5 reached at 20.5 s


def test_traj_example_rule_and_figure(plt, tmp_path):
    rt, traj = _traj_example_data()
    ex = F.pick_traj_example(rt, traj, CFG)
    info = ex["info"]
    assert info["ice_run"] == "ICE_VehA_dip_min" and info["ice_gap"] == "min"          # first vehicle, first gap
    assert info["ice_leader_min_ms"] == pytest.approx(24.1 - 7.7, abs=0.01)
    assert info["ice_min_time_s"] == pytest.approx(460.0, abs=0.2)
    assert info["ev_run"] == "EV_IONIQ5_Short_55to35_r1"                               # 55 mph, 35 mph, Short, r1
    assert (info["ev_v_free_mph"], info["ev_v_reduced_mph"]) == (55.0, 35.0)
    assert info["ev_onset_s"] == pytest.approx(10.3, abs=0.11)
    for key in ("ev", "ice"):
        e = ex[key]
        assert e["t_rel"].min() >= -5.0 - 1e-6 and e["t_rel"].max() <= ex["ev"]["t_rel"].max() + 1e-6
        assert len(e["t_rel"]) == len(e["s"]) > 100
    assert ex["ev"]["t_rel"].min() == pytest.approx(-5.0, abs=1e-6)
    fig = F.traj_example_figure(plt, ex, CFG)
    _check_fig(fig, "single_width_in", "traj_example", tmp_path, "traj")
    assert [x.get_text() for x in fig.legends[0].get_texts()] == ["EV (IONIQ 5, Short gap)", "ICE (VehA, min gap)"]
    assert fig.axes[0].get_xlabel() == "time from start of leader braking (s)"


def test_schematics(plt, tmp_path):
    from matplotlib.patches import Rectangle

    from ema import protocol
    cfg_e = config.load_yaml("evaluation.yaml")
    rows = [dict(run_id=f"{v}_{g}", powertrain=pt, vehicle=v, gap=g)
            for pt, vs, gs in (("EV", ["IONIQ 5"], ["Short", "Medium", "Long", "XLong"]),
                               ("EV", ["Model 3", "Polestar 2"], ["Short"]), ("ICE", ICES, ["min", "max"]))
            for v in vs for g in gs]
    rt = pd.DataFrame(rows).set_index("run_id")
    rt["task"] = rt[cfg_e["task_keys"]].astype(str).agg(" | ".join, axis=1)
    folds = protocol.make_folds(rt, cfg_e)
    fig = F.levels_schematic_figure(plt, folds, F.schematic_tasks(rt), CFG)
    _check_fig(fig, "single_width_in", "levels_schematic", tmp_path, "levels")
    ax = fig.axes[0]
    cells = [p for p in ax.patches if isinstance(p, Rectangle) and p.get_hatch() is not None]
    assert len(cells) == 1 + 4 + 4 + 6                  # first folds: IONIQ 5 Long; IONIQ 5; IONIQ 5; all EVs
    texts = [x.get_text() for x in ax.texts]
    assert {"16 folds", "9 folds", "3 folds", "1 fold"} <= set(texts)
    assert all(F.level_label(CFG, lv) in texts for lv in CFG["schematics"]["level_order"])
    assert [x.get_text() for x in fig.legends[0].get_texts()] == ["test tasks", "training tasks", "not used"]
    fig = F.methods_schematic_figure(plt, CFG)
    _check_fig(fig, "single_width_in", "methods_schematic", tmp_path, "methods")
    grouped = [m for g in CFG["schematics"]["method_groups"] for m in g["methods"]]
    assert sorted(grouped) == sorted(CFG["groups"]["all"])                      # every method exactly once
    body = " ".join(x.get_text() for x in fig.axes[0].texts)
    n_post = sum(CFG["methods"][m].get("design") == "post" for m in grouped)
    assert body.count(chr(8224)) == n_post == 6
    assert all(CFG["methods"][m]["label"] in body for m in grouped)
