"""Figures and tables of the paper (D034; rules D037; designs D040, D041, D043).

Every drawing function takes data that is already loaded (DataFrames, dicts) and returns a
matplotlib figure with exactly one axes, drawn at its final size; every table function
returns a DataFrame. scripts/make_figures.py loads the results and saves PDF + PNG figures
and CSV + LaTeX tables. Nothing here reads files, so everything can be tested on synthetic
data.
"""
import numpy as np
import pandas as pd

from ema import comparison

LEVEL_LABELS = {"L1_new_setting": "L1 new setting", "L2_new_vehicle": "L2 new vehicle",   # fallback only
                "L2_no_ice": "L2 EV only", "L3_cross_powertrain": "L3 cross-powertrain"}
LEVEL_SHORT = {"L1_new_setting": "L1", "L2_new_vehicle": "L2", "L2_no_ice": "L2 EV only",
               "L3_cross_powertrain": "L3"}
DESIGN_LABELS = {"pre": "pre-specified", "planned": "planned", "post": "post hoc"}
STATUS_SHORT = {"pre": "pre", "planned": "plan", "post": "post"}       # tab_all_methods; caption explains


# ---------------------------------------------------------------- style and output

def setup(cfg_fig):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = cfg_fig["style"]
    plt.rcParams.update({"font.family": s["font_family"], "font.serif": list(s["font_serif"]),
                         "font.size": s["font_size"], "axes.titlesize": s["font_size"],
                         "axes.labelsize": s["font_size"], "legend.fontsize": s["font_size"] - 1,
                         "xtick.labelsize": s["font_size"] - 1, "ytick.labelsize": s["font_size"] - 1,
                         "lines.linewidth": 1.0, "lines.markersize": 3.5, "mathtext.fontset": "stix",
                         "pdf.fonttype": 42, "savefig.bbox": "standard"})   # D037 (2): saved as drawn
    return plt


def new_figure(plt, cfg_fig, width_key, height_key):
    """One figure with one axes (D037 (1)) at the configured final size; constrained layout,
    so a legend below the axes (D046) is placed inside the figure."""
    st = cfg_fig["style"]
    return plt.subplots(figsize=(st[width_key], cfg_fig["heights_in"][height_key]), layout="constrained")


def legend_below(fig, cfg_fig, handles, ncol, title=None):
    """D046: every legend below the axes, never over the data."""
    st = cfg_fig["style"]
    return fig.legend(handles=handles, loc="outside lower center", ncol=ncol, frameon=False,
                      fontsize=st["font_size"] - 1.5, title=title, title_fontsize=st["font_size"] - 1,
                      columnspacing=1.0, handlelength=2.2)


def save(fig, out_dir, name, dpi=300):
    """PDF and PNG; the PDF without a creation date, so equal figures give equal files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", metadata={"CreationDate": None})
    fig.savefig(out_dir / f"{name}.png", dpi=dpi)


def _latex_cell(v, fmt, col="", raw=False):
    """One LaTeX cell: numbers with a math minus, NaN as --, p-values with four decimals,
    ranges like 17-99 with an en dash; text escaped unless raw."""
    import re
    bs = chr(92)
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return ("$-$" if v < 0 else "") + str(abs(int(v)))
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "--"
        f = "{:.4f}" if str(col).startswith("p ") or str(col) == "p" else fmt
        out = f.format(abs(v))
        return ("$-$" + out) if v < 0 and float(out) != 0 else out
    s = str(v)
    if raw:
        return s
    if re.fullmatch(r"[0-9]+-[0-9]+", s):
        return s.replace("-", "--")
    return (s.replace(bs, bs + "textbackslash{}").replace("_", bs + "_").replace("%", bs + "%")
            .replace("&", bs + "&").replace("#", bs + "#"))


def latex_table(df, caption, label, float_fmt="{:.2f}", bold=None, note=None, wide=False,
                header=None, groups=None, raw_cols=(), row_groups=None, size="footnotesize", colsep=4):
    """A booktabs LaTeX table without extra dependencies.
    bold: DataFrame of booleans of the same shape (cells in bold); note: text under the table;
    wide: table* (two columns); header: LaTeX column names (raw) instead of df.columns;
    groups: [(label, n_columns), ...], a header row above the column names (empty label: no
    group); raw_cols: columns whose text is LaTeX already; row_groups: [(label, n_rows), ...],
    an italic label row before each block of rows; size: font size of the table (IEEE tables:
    footnotesize); colsep: tabcolsep in pt. A header cell with a line break (chr(10)) is set
    on two lines.
    Columns whose name is "p" or starts with "p " get four decimals."""
    bs, nl = chr(92), chr(10)
    env = "table*" if wide else "table"
    import re

    def numeric_like(c):                     # numbers, and strings such as 7/9, 17-99, 300
        return pd.api.types.is_numeric_dtype(df[c]) or all(
            re.fullmatch(r"[0-9./ -]+", str(x)) for x in df[c])

    def stack(text):                         # a cell on two lines where the text has a line break
        parts = [_latex_cell(x, float_fmt) for x in text.split(nl)]
        return parts[0] if len(parts) == 1 else (bs + "begin{tabular}[b]{@{}c@{}}" + (bs + bs).join(parts)
                                                 + bs + "end{tabular}")

    cols = "".join("r" if numeric_like(c) else "l" for c in df.columns)
    end = " " + bs + bs
    lines = [bs + "begin{" + env + "}[t]", bs + "centering"]
    if size:
        lines.append(bs + size)
    if colsep is not None:
        lines.append(bs + "setlength{" + bs + "tabcolsep}{" + f"{colsep:g}pt" + "}")
    lines += [bs + "caption{" + caption + "}", bs + "label{" + label + "}",
              bs + "begin{tabular}{" + cols + "}", bs + "toprule"]
    if groups:
        if sum(n for _, n in groups) != len(df.columns):
            raise ValueError("groups do not cover the columns")
        cells, rules, c0 = [], [], 1
        for g, n in groups:
            if g:
                cells.append(bs + "multicolumn{" + str(n) + "}{c}{" + stack(g) + "}")
                rules.append(bs + "cmidrule(lr){" + f"{c0}-{c0 + n - 1}" + "}")
            else:
                cells += [""] * n
            c0 += n
        lines += [" & ".join(cells) + end, " ".join(rules)]
    names = header if header is not None else [_latex_cell(c, float_fmt) for c in df.columns]
    names = [bs + "begin{tabular}[b]{@{}c@{}}" + n.replace(nl, bs + bs) + bs + "end{tabular}" if nl in n else n
             for n in names]
    if len(names) != len(df.columns):
        raise ValueError("header does not match the columns")
    lines += [" & ".join(names) + end, bs + "midrule"]
    starts = {}
    if row_groups:
        if sum(n for _, n in row_groups) != len(df):
            raise ValueError("row groups do not cover the rows")
        r0 = 0
        for g, n in row_groups:
            starts[r0] = g
            r0 += n
    for i, row in enumerate(df.itertuples(index=False)):
        if i in starts:
            if i > 0:
                lines.append(bs + "addlinespace")
            lines.append(bs + "multicolumn{" + str(len(df.columns)) + "}{l}{" + bs + "textit{"
                         + _latex_cell(starts[i], float_fmt) + "}}" + end)
        cells = []
        for j, (v, c) in enumerate(zip(row, df.columns)):
            cell = _latex_cell(v, float_fmt, c, raw=c in raw_cols)
            if bold is not None and bool(bold.iloc[i, j]):
                cell = bs + "textbf{" + cell + "}"
            cells.append(cell)
        lines.append(" & ".join(cells) + end)
    lines += [bs + "bottomrule", bs + "end{tabular}"]
    if note:
        lines += [bs + "par" + bs + "smallskip", bs + "parbox{" + bs + "linewidth}{" + bs + "footnotesize " + note + "}"]
    lines.append(bs + "end{" + env + "}")
    return nl.join(lines) + nl


def save_table(df, out_dir, name, caption, label, **latex):
    """CSV (values as computed) and LaTeX (latex_table options in **latex)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}.csv", index=False, lineterminator=chr(10))
    ldf = latex.pop("latex_df", df)
    with open(out_dir / f"{name}.tex", "w", encoding="utf-8", newline="") as f:
        f.write(latex_table(ldf, caption, label, **latex))


def _log_x_lengths(ax, lengths):
    from matplotlib.ticker import NullLocator
    ax.set_xscale("log")
    ax.set_xticks(lengths)
    ax.set_xticklabels([str(int(L)) for L in lengths])
    ax.xaxis.set_minor_locator(NullLocator())


def _log_ticks(ax, axis, candidates):
    """Plain-number ticks on a log axis, chosen from candidates inside the current limits."""
    from matplotlib.ticker import FixedLocator, NullLocator
    lo, hi = ax.get_ylim() if axis == "y" else ax.get_xlim()
    ticks = [t for t in candidates if lo <= t <= hi]
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(FixedLocator(ticks))
    a.set_minor_locator(NullLocator())
    (ax.set_yticklabels if axis == "y" else ax.set_xticklabels)([f"{t:g}" for t in ticks])


def level_label(cfg_fig, level):
    """Reader name of a level (D047), from config level_labels."""
    return cfg_fig.get("level_labels", {}).get(level, LEVEL_LABELS.get(level, level))


def _pt_style(cfg_fig, pt):
    return cfg_fig["powertrains"][pt]


def parse_task(task):
    """'EV | IONIQ 5 | Short' -> ('EV', 'IONIQ 5', 'Short')."""
    pt, veh, gap = [x.strip() for x in str(task).split("|")]
    return pt, veh, gap


def gap_rank(gap, cfg_fig):
    """Position of a gap setting in config gap_order (unknown settings sort last)."""
    order = list(cfg_fig.get("gap_order", []))
    return order.index(gap) if gap in order else len(order)


def spread_labels(y, min_gap):
    """Label positions close to y (any order) with at least min_gap between neighbours,
    keeping their order; used to keep end-of-line labels from overlapping."""
    y = np.asarray(y, dtype=float)
    order = np.argsort(y)
    pos = y[order].copy()
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + min_gap)
    pos -= (pos - y[order]).mean()                  # re-centre the group
    for i in range(1, len(pos)):                    # keep the spacing after re-centring
        pos[i] = max(pos[i], pos[i - 1] + min_gap)
    out = np.empty_like(pos)
    out[order] = pos
    return out


# ---------------------------------------------------------------- data (tab_data)

def data_overview(rt):
    """tab_data: one row per vehicle of the evaluation run table, and a total."""
    g = rt.reset_index().groupby(["powertrain", "vehicle"])
    out = g.agg(gap_settings=("gap", "nunique"), runs=("run_id", "size"),
                configurations=("config_id", "nunique"), hours=("duration_s", "sum"),
                stop_runs=("is_stop_run", "sum")).reset_index()
    out["hours"] = out["hours"] / 3600.0
    out["stop_runs"] = out["stop_runs"].astype(int)
    total = dict(powertrain="all", vehicle=f"{len(out)} vehicles", gap_settings=out["gap_settings"].sum(),
                 runs=out["runs"].sum(), configurations=out["configurations"].sum(),
                 hours=out["hours"].sum(), stop_runs=out["stop_runs"].sum())
    t = pd.concat([out.sort_values(["powertrain", "vehicle"]), pd.DataFrame([total])], ignore_index=True)
    return t.rename(columns={"powertrain": "Powertrain", "vehicle": "Vehicle", "gap_settings": "Gap settings",
                             "runs": "Runs", "configurations": "Configurations", "hours": "Hours",
                             "stop_runs": "Stop runs"})


# ---------------------------------------------------------------- fig_traj_example (D043, D055)

def leader_braking_onset(t, v_l, lookback_s, tol):
    """D055 (4): (onset index, index of the lowest leader speed). The onset is the last time
    before the minimum at which the leader speed is at least the highest speed of the
    lookback_s before the minimum minus tol."""
    t, v_l = np.asarray(t, float), np.asarray(v_l, float)
    i_min = int(np.argmin(v_l))
    ref = float(v_l[(t >= t[i_min] - lookback_s) & (t <= t[i_min])].max())
    ok = np.flatnonzero(v_l[: i_min + 1] >= ref - tol)
    return int(ok[-1]), i_min


def pick_traj_example(rt, traj, cfg_fig):
    """D055: the ICE and the EV example run of fig_traj_example, chosen from the leader and
    the settings only (never from the follower). rt: run table (index run_id; powertrain,
    vehicle, gap, profile, v_free_mph, v_reduced_mph, is_stop_run); traj: run_id, t, v_l, s.
    Returns dict(ev=..., ice=..., info=...); ev and ice hold t_rel (time from the start of
    leader braking), s and v_l over the plotted span."""
    c = cfg_fig["traj_example"]
    mph = 0.44704
    r = rt[~rt["is_stop_run"].astype(bool) & (rt["profile"] == c["profile"])]

    def series(rid):
        d = traj[traj["run_id"] == rid].sort_values("t")
        return d["t"].to_numpy(float), d["v_l"].to_numpy(float), d["s"].to_numpy(float)

    # (1) ICE: first ICE vehicle, first gap setting in gap order; (2) its lowest leader speed
    ice = r[r["powertrain"] == "ICE"].reset_index()
    ice = ice.assign(rank=[gap_rank(g, cfg_fig) for g in ice["gap"]]).sort_values(["vehicle", "rank", "run_id"])
    ice_row = ice.iloc[0]
    t_i, vl_i, s_i = series(ice_row["run_id"])
    on_i, min_i = leader_braking_onset(t_i, vl_i, c["lookback_s"], c["tol_ms"])
    cruise = float(np.median(vl_i))
    # (3) EV: the EV with most gap settings, free speed closest to the ICE cruise speed,
    # reduced speed closest to the ICE minimum, first gap setting, first run id
    ev = r[r["powertrain"] == "EV"].reset_index()
    n_gaps = ev.groupby("vehicle")["gap"].nunique()
    veh = sorted(n_gaps.index[n_gaps == n_gaps.max()])[0]
    ev = ev[ev["vehicle"] == veh]
    vf = min(sorted(ev["v_free_mph"].unique()), key=lambda x: abs(x * mph - cruise))
    ev = ev[ev["v_free_mph"] == vf]
    vr = min(sorted(ev["v_reduced_mph"].unique()), key=lambda x: abs(x * mph - vl_i[min_i]))
    ev = ev[ev["v_reduced_mph"] == vr]
    ev = ev.assign(rank=[gap_rank(g, cfg_fig) for g in ev["gap"]]).sort_values(["rank", "run_id"])
    ev_row = ev.iloc[0]
    t_e, vl_e, s_e = series(ev_row["run_id"])
    on_e, min_e = leader_braking_onset(t_e, vl_e, c["lookback_s"], c["tol_ms"])
    # (5) span: from -pre_s to the end of the EV run, the ICE run cut to the same span
    te_rel, ti_rel = t_e - t_e[on_e], t_i - t_i[on_i]
    lo, hi = -float(c["pre_s"]), float(te_rel[-1])
    eps = 1e-6                                          # sample times are multiples of 0.1 s up to rounding
    me, mi = te_rel >= lo - eps, (ti_rel >= lo - eps) & (ti_rel <= hi + eps)
    out = {}
    for key, row, tr_, vl, s, m, on, i_min in (("ev", ev_row, te_rel, vl_e, s_e, me, on_e, min_e),
                                              ("ice", ice_row, ti_rel, vl_i, s_i, mi, on_i, min_i)):
        out[key] = dict(run_id=row["run_id"], powertrain=row["powertrain"], vehicle=row["vehicle"],
                        gap=row["gap"], t_rel=tr_[m], s=s[m], v_l=vl[m])
    out["info"] = dict(
        ice_run=ice_row["run_id"], ice_vehicle=ice_row["vehicle"], ice_gap=ice_row["gap"],
        ice_onset_s=round(float(t_i[on_i]), 2), ice_min_time_s=round(float(t_i[min_i]), 2),
        ice_leader_cruise_ms=round(cruise, 2), ice_leader_min_ms=round(float(vl_i[min_i]), 2),
        ev_run=ev_row["run_id"], ev_vehicle=veh, ev_gap=ev_row["gap"], ev_v_free_mph=float(vf),
        ev_v_reduced_mph=float(vr), ev_onset_s=round(float(t_e[on_e]), 2),
        ev_min_time_s=round(float(t_e[min_e]), 2), ev_leader_max_ms=round(float(vl_e.max()), 2),
        ev_leader_min_ms=round(float(vl_e[min_e]), 2), span_s=[lo, round(hi, 2)])
    return out


def traj_example_figure(plt, ex, cfg_fig):
    """fig_traj_example (D043, D055): gap against the time from the start of leader braking
    of the EV and the ICE example run; leader speeds go into the caption."""
    fig, ax = new_figure(plt, cfg_fig, "single_width_in", "traj_example")
    handles = []
    for key in ("ev", "ice"):
        e = ex[key]
        st = _pt_style(cfg_fig, e["powertrain"])
        h, = ax.plot(e["t_rel"], e["s"], color=st["color"], ls=st["ls"], lw=1.0,
                     label=f"{e['powertrain']} ({e['vehicle']}, {e['gap']} gap)")
        handles.append(h)
    ax.axvline(0.0, color="0.5", lw=0.5, ls=":", zorder=1)
    ax.set_xlabel("time from start of leader braking (s)")
    ax.set_ylabel("gap (m)")
    ax.grid(alpha=0.3, lw=0.4)
    legend_below(fig, cfg_fig, handles, ncol=2)
    return fig


# ---------------------------------------------------------------- schematics (D056)

def schematic_tasks(rt):
    """Task table of the schematics: index task; powertrain, vehicle, gap."""
    return rt.groupby("task")[["powertrain", "vehicle", "gap"]].first().sort_index()


def levels_schematic_figure(plt, folds, tasks, cfg_fig):
    """fig_levels_schematic (D056): one strip per level for its first fold, one column per
    vehicle (EVs first), one cell per gap setting: test hatched, training grey, unused white;
    the number of folds per level; below, the steps inside a test task. folds: list of
    dict(fold_id, level, test, train) as made by ema.protocol.make_folds."""
    from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
    st = cfg_fig["style"]
    sc = cfg_fig["schematics"]
    fs = st["font_size"] - 1.5
    ev_col, ice_col = _pt_style(cfg_fig, "EV")["color"], _pt_style(cfg_fig, "ICE")["color"]
    fig, ax = new_figure(plt, cfg_fig, "single_width_in", "levels_schematic")
    vehicles = (sorted(tasks.loc[tasks["powertrain"] == "EV", "vehicle"].unique())
                + sorted(tasks.loc[tasks["powertrain"] == "ICE", "vehicle"].unique()))
    gaps = {v: sorted(tasks.loc[tasks["vehicle"] == v, "gap"], key=lambda g: gap_rank(g, cfg_fig)) for v in vehicles}
    task_of = {(r.vehicle, r.gap): t for t, r in tasks.iterrows()}
    w, sep, h = 1.0, 0.18, 0.55
    x0 = {v: i * (w + sep) for i, v in enumerate(vehicles)}
    x_end = x0[vehicles[-1]] + w
    levels = [lv for lv in sc["level_order"] if any(f["level"] == lv for f in folds)]
    row_h = 1.25
    y_top = 0.0
    for i, v in enumerate(vehicles):                                   # vehicle names on top
        pt = tasks.loc[tasks["vehicle"] == v, "powertrain"].iloc[0]
        ax.text(x0[v] + w / 2, y_top + 0.15, v, ha="center", va="bottom", rotation=90, fontsize=fs,
                color=ev_col if pt == "EV" else ice_col)
    for k, lv in enumerate(levels):
        fl = [f for f in folds if f["level"] == lv]
        f0 = fl[0]
        y = y_top - (k + 1) * row_h
        ax.text(0.0, y + h + 0.12, level_label(cfg_fig, lv), ha="left", va="bottom", fontsize=fs)
        ax.text(x_end, y + h + 0.12, f"{len(fl)} fold{'s' if len(fl) != 1 else ''}", ha="right",
                va="bottom", fontsize=fs, style="italic")
        for v in vehicles:
            n = len(gaps[v])
            for j, g in enumerate(gaps[v]):
                t = task_of[(v, g)]
                xx = x0[v] + j * w / n
                if t in f0["test"]:
                    fc, hatch, ec = "white", "/////", "k"
                elif t in f0["train"]:
                    fc, hatch, ec = "0.72", None, "0.35"
                else:
                    fc, hatch, ec = "white", None, "0.6"
                ax.add_patch(Rectangle((xx, y), w / n, h, fc=fc, ec=ec, lw=0.5, hatch=hatch))
    # steps inside a test task
    yb = y_top - (len(levels) + 1) * row_h - 0.35
    nl = chr(10)
    boxes = sc["steps"]
    bw = (x_end - 2 * 0.35) / len(boxes)
    for i, text in enumerate(boxes):
        bx = i * (bw + 0.35)
        ax.add_patch(Rectangle((bx, yb - 0.95), bw, 1.75, fc="0.95", ec="0.35", lw=0.5))
        ax.text(bx + bw / 2, yb - 0.075, text.replace("|", nl), ha="center", va="center", fontsize=fs)
        if i:
            ax.add_patch(FancyArrowPatch((bx - 0.33, yb - 0.075), (bx - 0.02, yb - 0.075),
                                         arrowstyle="-|>", mutation_scale=9, lw=0.8, color="k"))
    ax.text(0.0, yb + 0.92, sc["steps_title"], ha="left", va="bottom", fontsize=fs)
    ax.set_xlim(-0.05, x_end + 0.05)
    ax.set_ylim(yb - 1.05, y_top + 1.9)
    ax.axis("off")
    handles = [Patch(fc="white", ec="k", hatch="/////", lw=0.5, label="test tasks"),
               Patch(fc="0.72", ec="0.35", lw=0.5, label="training tasks"),
               Patch(fc="white", ec="0.6", lw=0.5, label="not used")]
    legend_below(fig, cfg_fig, handles, ncol=3)
    return fig


def methods_schematic_figure(plt, cfg_fig):
    """fig_methods_schematic (D056): the method families from most to least physics, one box
    each with the reader names of its methods; post hoc designs marked with a dagger."""
    from matplotlib.patches import FancyArrowPatch, Rectangle
    st = cfg_fig["style"]
    fs = st["font_size"] - 1.5
    groups = cfg_fig["schematics"]["method_groups"]
    fig, ax = new_figure(plt, cfg_fig, "single_width_in", "methods_schematic")
    dagger = chr(8224)
    y, gap_y, x_box, w_box = 0.0, 0.35, 0.0, 9.0
    tops = []
    for g in groups:
        names = [cfg_fig["methods"][m]["label"] + (dagger if cfg_fig["methods"][m].get("design") == "post" else "")
                 for m in g["methods"]]
        lines = ["; ".join(names[i:i + g.get("per_line", 3)]) for i in range(0, len(names), g.get("per_line", 3))]
        hgt = 0.95 + 0.62 * len(lines)
        ax.add_patch(Rectangle((x_box, y - hgt), w_box, hgt, fc="0.95", ec="0.35", lw=0.5))
        ax.text(x_box + 0.15, y - 0.2, g["title"], ha="left", va="top", fontsize=fs, weight="bold")
        for i, ln in enumerate(lines):
            ax.text(x_box + 0.15, y - 0.82 - 0.62 * i, ln, ha="left", va="top", fontsize=fs)
        tops.append((y, y - hgt))
        y -= hgt + gap_y
    y_bottom = y + gap_y
    ax.add_patch(FancyArrowPatch((w_box + 0.45, 0.0), (w_box + 0.45, y_bottom), arrowstyle="-|>",
                                 mutation_scale=7, lw=0.7, color="k"))
    ax.text(w_box + 0.75, y_bottom / 2, cfg_fig["schematics"]["methods_arrow"], rotation=270, ha="center",
            va="center", fontsize=fs)
    ax.set_xlim(-0.05, w_box + 1.05)
    ax.set_ylim(y_bottom - 0.05, 0.05)
    ax.axis("off")
    return fig


# ---------------------------------------------------------------- curves (fig_q1_l2, fig_ai_methods)

def mean_with_interval(values, level, lengths, cfg_stat, powertrain=None, name=""):
    """Per length: mean over vehicles and the bootstrap interval of that mean (D035 unit)."""
    v = comparison.at_lengths(values, lengths)
    v = v[v["level"] == level]
    if powertrain:
        v = v[v["powertrain"] == powertrain]
    rows = []
    for L in lengths:
        x = v.loc[v["length_s"] == float(L), "value"].dropna().to_numpy()
        if len(x) == 0:
            continue
        seed = int(np.random.SeedSequence(comparison._seed(cfg_stat["seed"], "mean", name, level,
                                                           str(powertrain), str(float(L))))
                   .generate_state(1)[0])
        m, lo, hi, _ = comparison.bootstrap(x, cfg_stat["n_boot"], cfg_stat["ci"], seed)
        rows.append(dict(length_s=float(L), mean=m, lo=lo, hi=hi, n_vehicles=len(x)))
    return pd.DataFrame(rows)


def curves_figure(plt, values, methods, level, lengths, cfg_fig, cfg_stat, height_key,
                  log_y=False, reference=None, intervals=True):
    """Vehicle-level mean spacing RMSE against context length, one line per method, with
    vehicle-unit 95 % intervals where the level has enough vehicles (D035) unless intervals
    is False (fig_ai_methods, D045). reference: a method drawn as a grey dashed line
    without interval. Methods with open: true in the config get open markers."""
    fig, ax = new_figure(plt, cfg_fig, "single_width_in", height_key)
    names = [m for m in methods if m in values]
    for k, name in enumerate(names):
        st = cfg_fig["methods"][name]
        m = mean_with_interval(values[name], level, lengths, cfg_stat, None, name)
        if m.empty:
            continue
        x = m["length_s"] * (1.0 + 0.035 * (k - (len(names) - 1) / 2))       # small dodge
        show = intervals and (m["n_vehicles"] >= cfg_stat["min_vehicles_for_intervals"]).all()
        yerr = np.vstack([m["mean"] - m["lo"], m["hi"] - m["mean"]]) if show else None
        ax.errorbar(x, m["mean"], yerr=yerr, color=st["color"], ls=st["ls"], marker=st["marker"],
                    mfc="white" if st.get("open") else st["color"], mec=st["color"],
                    label=st["label"], capsize=1.5, elinewidth=0.6, zorder=3)
    if reference and reference in values:
        m = mean_with_interval(values[reference], level, lengths, cfg_stat, None, reference)
        ax.plot(m["length_s"], m["mean"], color="0.45", ls="--", lw=0.9, zorder=2,
                label=cfg_fig["methods"][reference]["label"] + " (reference)")
    _log_x_lengths(ax, lengths)
    if log_y:
        ax.set_yscale("log")
        _log_ticks(ax, "y", (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 200, 500))
    ax.set_xlabel(cfg_fig["x_label"])
    ax.set_ylabel("spacing RMSE (m)")
    ax.grid(alpha=0.3, lw=0.4)
    legend_below(fig, cfg_fig, ax.get_legend_handles_labels()[0], ncol=2)
    return fig


# ---------------------------------------------------------------- tab_q1

def confirmatory_table(intervals, cfg_fig):
    """tab_q1: the pre-specified Q1 comparisons with Holm-corrected p-values. Returns the CSV
    table and the LaTeX options (one block of rows per comparison, D049)."""
    c = intervals[intervals["kind"] == "confirmatory"].copy()
    lab = {k: v["label"] for k, v in cfg_fig["methods"].items()}
    comp = [f"{lab.get(a, a)} vs {lab.get(b, b)}" for a, b in zip(c["method"], c["versus"])]
    out = pd.DataFrame({"comparison": comp, "observed driving (s)": c["length_s"].astype(int).astype(str),
                        "mean diff. (m)": c["mean_diff"], "CI low (m)": c["ci_lo"], "CI high (m)": c["ci_hi"],
                        "vehicles better": c["n_better"].astype(str) + "/" + c["n_vehicles"].astype(str),
                        "p (Holm)": c["p_holm"]}).reset_index(drop=True)
    blocks = [(k, len(list(g))) for k, g in __import__("itertools").groupby(out["comparison"])]
    latex = dict(latex_df=out.drop(columns="comparison"), row_groups=blocks,
                 header=["Observed" + chr(10) + "driving (s)", "Mean" + chr(10) + "diff. (m)", "Low", "High",
                         "Vehicles" + chr(10) + "better", "$p$" + chr(10) + "(Holm)"],
                 groups=[("", 2), ("95% CI (m)", 2), ("", 2)])
    return out, latex


# ---------------------------------------------------------------- fig_method_vehicle

def vehicle_heatmap(plt, values, methods, level, length, cfg_fig):
    """fig_method_vehicle (D043 (3)): methods x vehicles at one context length; the values are
    written in the cells; logarithmic colour scale with a colour bar on the right (D061, the
    only figure with a second axes); no title."""
    rows, labels, vehicles = [], [], None
    for name in methods:
        if name not in values:
            continue
        v = comparison.at_lengths(values[name], [float(length)])
        v = v[(v["level"] == level) & (v["length_s"] == float(length))]
        if v.empty:
            continue
        v = v.assign(col=v["powertrain"] + " " + v["vehicle"]).set_index("col")["value"]
        vehicles = sorted(v.index, key=lambda c: (c.split()[0], c)) if vehicles is None else vehicles
        rows.append(v.reindex(vehicles).to_numpy())
        labels.append(cfg_fig["methods"][name]["label"])
    M = np.array(rows, dtype=float)
    fig, ax = new_figure(plt, cfg_fig, "double_width_in", "method_vehicle")
    from matplotlib.colors import LogNorm
    im = ax.imshow(M, aspect="auto", cmap="viridis_r",
                   norm=LogNorm(vmin=np.nanmin(M), vmax=np.nanmax(M)))     # same colours as log(M), D061
    ax.set_xticks(range(len(vehicles)))
    ax.set_xticklabels([c.split(" ", 1)[1] + " (" + c.split()[0] + ")" for c in vehicles])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.tick_params(length=0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                r, g, b, _ = im.cmap(im.norm(M[i, j]))
                lum = 0.2126 * r + 0.7152 * g + 0.0722 * b          # D048: text colour by cell brightness
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                        fontsize=cfg_fig["style"]["font_size"] - 2, color="w" if lum < 0.45 else "k")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, aspect=30)          # D061
    lo, hi = np.nanmin(M), np.nanmax(M)
    ticks = [t for t in (1, 2, 3, 5, 10, 20, 30, 50, 100, 200) if lo <= t <= hi]
    cb.ax.minorticks_off()
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:g}" for t in ticks])
    cb.ax.tick_params(labelsize=cfg_fig["style"]["font_size"] - 1, length=2)
    cb.set_label("spacing RMSE (m), logarithmic scale", fontsize=cfg_fig["style"]["font_size"] - 1)
    cb.outline.set_linewidth(0.5)
    return fig


# ---------------------------------------------------------------- fig_idm_params, tab_idm_params

def idm_task_params(idm_runs, cfg_fig, names=("v0", "T", "s0", "a", "b")):
    """One row per task from the diagnostic-C runs table (cross-configuration fits):
    powertrain, vehicle, gap and the IDM parameters, sorted by powertrain, vehicle, gap."""
    d = idm_runs[idm_runs["floor"] == "cross_config"]
    cols = [f"idm_{k}" for k in names]
    per = d.groupby("task")[cols]
    if (per.nunique() > 1).any().any():
        raise ValueError("a task has more than one cross-configuration fit")
    t = per.first().reset_index()
    parts = t["task"].map(parse_task)
    t["powertrain"], t["vehicle"], t["gap"] = parts.str[0], parts.str[1], parts.str[2]
    t["gap_rank"] = t["gap"].map(lambda g: gap_rank(g, cfg_fig))
    t = t.rename(columns={f"idm_{k}": k for k in names})
    return (t.sort_values(["powertrain", "vehicle", "gap_rank"])
             [["task", "powertrain", "vehicle", "gap", "gap_rank", *names]].reset_index(drop=True))


def idm_params_figure(plt, params, cfg_fig, bounds):
    """fig_idm_params (D048, replaces the T-s0 scatter of D037): desired time gap T per task,
    one row per vehicle (EVs on top), the gap settings of a vehicle joined in gap order and
    named; T is the parameter not pinned at a bound (s0, v0, b: tab_idm_params)."""
    st = cfg_fig["style"]
    fig, ax = new_figure(plt, cfg_fig, "single_width_in", "idm_params")
    vehicles = (params[["powertrain", "vehicle"]].drop_duplicates()
                .sort_values(["powertrain", "vehicle"]).itertuples(index=False))
    vehicles = list(vehicles)
    n = len(vehicles)
    for i, (pt, veh) in enumerate(vehicles):
        s = _pt_style(cfg_fig, pt)
        g = params[(params["powertrain"] == pt) & (params["vehicle"] == veh)].sort_values("gap_rank")
        y = n - 1 - i
        ax.plot(g["T"], [y] * len(g), color=s["color"], ls=s["ls"], lw=0.7, marker=s["marker"], ms=4,
                mfc=s["color"] if pt == "EV" else "white", mec=s["color"], zorder=3)
        for _, r in g.iterrows():
            ax.annotate(r["gap"], (r["T"], y), xytext=(0, 4), textcoords="offset points", ha="center",
                        va="bottom", fontsize=st["font_size"] - 2.5)
    lo, hi = bounds["T"]
    for b_ in (lo, hi):
        if abs(params["T"] - b_).min() <= 0.05 * (hi - lo):
            ax.axvline(b_, color="0.6", ls=":", lw=0.6, zorder=1)
    ax.set_yticks(range(n))
    ax.set_yticklabels([veh for _, veh in vehicles][::-1])
    ax.set_ylim(-0.6, n - 0.2)
    ax.margins(x=0.08)
    handles = [plt.Line2D([], [], color=_pt_style(cfg_fig, pt)["color"], ls=_pt_style(cfg_fig, pt)["ls"],
                          marker=_pt_style(cfg_fig, pt)["marker"],
                          mfc=_pt_style(cfg_fig, pt)["color"] if pt == "EV" else "white", label=pt)
               for pt in ("EV", "ICE") if pt in set(params["powertrain"])]
    legend_below(fig, cfg_fig, handles, ncol=2)
    ax.set_xlabel("desired time gap $T$ (s)")
    ax.grid(alpha=0.3, lw=0.4, axis="x")
    return fig


def idm_params_table(params, bounds, rel_tol=0.01):
    """tab_idm_params (D037, D049): the five free IDM parameters of the tasks; 'at bound' lists
    the parameters within rel_tol of the bound range from a bound. Returns (table, LaTeX
    options)."""
    names = ["v0", "T", "s0", "a", "b"]
    at = []
    for _, r in params.iterrows():
        hit = [k for k in names
               if min(abs(r[k] - bounds[k][0]), abs(r[k] - bounds[k][1]))
               <= rel_tol * (bounds[k][1] - bounds[k][0])]
        at.append(", ".join(hit) if hit else "-")
    t = pd.DataFrame({"Powertrain": params["powertrain"], "Vehicle": params["vehicle"],
                      "Gap": params["gap"], "v0 (m/s)": params["v0"], "T (s)": params["T"],
                      "s0 (m)": params["s0"], "a (m/s2)": params["a"], "b (m/s2)": params["b"],
                      "at bound": at})
    math = {"v0": "$v_0$", "T": "$T$", "s0": "$s_0$", "a": "$a$", "b": "$b$"}
    ldf = t.assign(**{"at bound": [", ".join(math[k] for k in x.split(", ")) if x != "-" else "--"
                                   for x in t["at bound"]]})
    units = {"v0": "m/s", "T": "s", "s0": "m", "a": "m/s$^2$", "b": "m/s$^2$"}
    bnd = "; ".join(f"{math[k]} {bounds[k][0]:g}--{bounds[k][1]:g} {units[k]}" for k in names)
    nl = chr(10)
    ldf = ldf.assign(Vehicle=ldf["Vehicle"] + " (" + ldf["Powertrain"] + ")").drop(columns="Powertrain")
    latex = dict(latex_df=ldf, raw_cols=("at bound",),
                 header=["Vehicle", "Gap", "$v_0$" + nl + "(m/s)", "$T$" + nl + "(s)", "$s_0$" + nl + "(m)",
                         "$a$" + nl + "(m/s$^2$)", "$b$" + nl + "(m/s$^2$)", "At" + nl + "bound"],
                 note="Bounds of the fit: " + bnd + ".", colsep=3)
    return t, latex


# ---------------------------------------------------------------- per-vehicle differences

def split_differences(values, a, b, level, lengths):
    """Per vehicle and length: d = value(a) - value(b) (negative: a better)."""
    return comparison.paired_differences(values[a], values[b], level, lengths)


def split_figure(plt, diff, cfg_fig, title, lengths):
    """fig_split_prior, fig_split_validation (D037), fig_ablation_pretraining (D040), layout of
    D045: one line per vehicle (EV warm solid filled, ICE cool dashed open), zero line, a
    three-column legend below the axes whose title names the difference; no intervals
    (D035)."""
    fig, ax = new_figure(plt, cfg_fig, "single_width_in", "split")
    ax.axhline(0.0, color="0.3", lw=0.6, zorder=1)
    handles = []
    for pt in ("EV", "ICE"):
        s = _pt_style(cfg_fig, pt)
        vehs = sorted(diff.loc[diff["powertrain"] == pt, "vehicle"].unique())
        for i, veh in enumerate(vehs):
            g = diff[(diff["powertrain"] == pt) & (diff["vehicle"] == veh)].sort_values("length_s")
            c = s["palette"][i % len(s["palette"])]
            mk = s["vehicle_markers"][i % len(s["vehicle_markers"])]
            (h,) = ax.plot(g["length_s"], g["d"], color=c, ls=s["ls"], marker=mk, ms=3, lw=0.9,
                           mfc=c if pt == "EV" else "white", mec=c, zorder=3, label=f"{veh} ({pt})")
            handles.append(h)
    _log_x_lengths(ax, lengths)
    ax.set_xlabel(cfg_fig["x_label"])
    ax.set_ylabel(chr(916) + " spacing RMSE (m)")
    ax.grid(alpha=0.3, lw=0.4)
    legend_below(fig, cfg_fig, handles, ncol=3, title=title)
    return fig


# ---------------------------------------------------------------- fig_selected_steps

def fold_test_info(runs):
    """fold_id, test_powertrain, test_label (the vehicle, or 'all EVs' / 'all ICEs' when a
    fold tests several vehicles of one powertrain), from a runs.csv table."""
    r = runs[["fold_id", "task"]].drop_duplicates()
    parts = r["task"].map(parse_task)
    r = r.assign(powertrain=parts.str[0], vehicle=parts.str[1])
    rows = []
    for fid, g in r.groupby("fold_id"):
        pts, vehs = sorted(set(g["powertrain"])), sorted(set(g["vehicle"]))
        label = vehs[0] if len(vehs) == 1 else f"all {pts[0]}s" if len(pts) == 1 else "all"
        rows.append(dict(fold_id=fid, test_powertrain="/".join(pts), test_label=label))
    return pd.DataFrame(rows)


def selected_steps(states_ft, states_lovo, fold_info):
    """One row per LOVO fold: level, test powertrain, test label, step chosen by the
    original validation (m4a_finetuned best_step) and by LOVO stage 1 (S)."""
    a = states_ft.set_index("fold_id")["best_step"].rename("step_original")
    b = states_lovo.set_index("fold_id")[["level", "best_step"]].rename(columns={"best_step": "step_lovo"})
    out = b.join(a, how="left").join(fold_info.set_index("fold_id"), how="left").reset_index()
    if out["step_original"].isna().any() or out["test_label"].isna().any():
        raise ValueError("missing m4a_finetuned state or fold information for a LOVO fold")
    return out


def selected_steps_figure(plt, steps, cfg_fig):
    """fig_selected_steps (D037): S of LOVO (y) against the original selected step (x),
    log-log with y = x; colour by test powertrain, marker by level, test label per point."""
    st = cfg_fig["style"]
    fig, ax = new_figure(plt, cfg_fig, "single_width_in", "selected_steps")
    lvl_mk = cfg_fig["level_markers"]
    offsets = cfg_fig.get("label_offsets", {}).get("selected_steps") or {}
    x = steps["step_original"].astype(float).to_numpy()
    y = steps["step_lovo"].astype(float).to_numpy()
    if (np.r_[x, y] <= 0).any():
        raise ValueError("a selected step is 0; the log axes of fig_selected_steps need a rule for it")
    lo, hi = min(x.min(), y.min()) * 0.6, max(x.max(), y.max()) * 1.6
    ax.plot([lo, hi], [lo, hi], color="0.5", ls="--", lw=0.6, zorder=1)
    placed = []
    for i in np.lexsort((x, y)):
        r = steps.iloc[i]
        s = _pt_style(cfg_fig, r["test_powertrain"])
        ax.plot(x[i], y[i], ls="none", marker=lvl_mk[r["level"]], ms=4.5, color=s["color"],
                mfc=s["color"] if r["test_powertrain"] == "EV" else "white", mec=s["color"], zorder=3)
        if r["fold_id"] in offsets:
            dx, dy = offsets[r["fold_id"]]
        else:
            n_near = sum(abs(np.log10(x[i] / x[j])) < 0.15 and abs(np.log10(y[i] / y[j])) < 0.15
                         for j in placed)
            dx, dy = 4, 2 + 7 * n_near
        placed.append(i)
        ax.annotate(r["test_label"], (x[i], y[i]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=st["font_size"] - 2, ha="left" if dx >= 0 else "right",
                    va="bottom" if dy >= 0 else "top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    _log_ticks(ax, "x", (10, 30, 50, 100, 300, 1000, 3000, 10000))
    _log_ticks(ax, "y", (10, 30, 50, 100, 300, 1000, 3000, 10000))
    ax.set_aspect("equal", adjustable="box")
    handles = [plt.Line2D([], [], ls="none", marker="o", color=_pt_style(cfg_fig, pt)["color"],
                          mfc=_pt_style(cfg_fig, pt)["color"] if pt == "EV" else "white",
                          label=f"{pt} test vehicle") for pt in ("EV", "ICE") if pt in set(steps["test_powertrain"])]
    handles += [plt.Line2D([], [], ls="none", marker=lvl_mk[l], color="0.3", mfc="white",
                           label=level_label(cfg_fig, l)) for l in lvl_mk if l in set(steps["level"])]
    legend_below(fig, cfg_fig, handles, ncol=2)
    ax.set_xlabel("fine-tuning steps chosen on all training vehicles")
    ax.set_ylabel("fine-tuning steps chosen on a held-out vehicle")
    ax.grid(alpha=0.3, lw=0.4)
    return fig


# ---------------------------------------------------------------- fig_physics_floors, tab_physics_floors

def _physics_models(tasks, cfg_fig):
    order = [m for m in cfg_fig.get("physics_model_order", []) if m in set(tasks["model"])]
    return order + sorted(set(tasks["model"]) - set(order))


def physics_floors_figure(plt, tasks, cfg_fig, floor="cross_config"):
    """fig_physics_floors (D041 (1)): task spacing RMSE (log axis) per physics model, EV and
    ICE side by side under each model, one point per task, medians as short bars."""
    labels = cfg_fig.get("physics_model_labels", {})
    models = _physics_models(tasks, cfg_fig)
    d = tasks[tasks["floor"] == floor]
    fig, ax = new_figure(plt, cfg_fig, "single_width_in", "physics_floors")
    for i, model in enumerate(models):
        for k, pt in enumerate(("EV", "ICE")):
            s = _pt_style(cfg_fig, pt)
            v = np.sort(d.loc[(d["model"] == model) & (d["powertrain"] == pt), "spacing_rmse_mean"].to_numpy())
            if not len(v):
                continue
            x0 = i + (k - 0.5) * 0.36
            rank = np.arange(len(v))                                  # D048: alternating offsets,
            xs = x0 + np.where(rank % 2, -1, 1) * ((rank + 1) // 2) * 0.03   # no ordering along x
            ax.plot(xs, v, ls="none", marker=s["marker"], ms=3.5, color=s["color"],
                    mfc=s["color"] if pt == "EV" else "white", mec=s["color"], zorder=3)
            ax.plot([x0 - 0.13, x0 + 0.13], [np.median(v)] * 2, color=s["color"], lw=1.4, zorder=4)
    ax.set_yscale("log")
    _log_ticks(ax, "y", (0.5, 1, 2, 3, 5, 10, 20, 50))
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([labels.get(m, m) for m in models])
    ax.set_xlim(-0.6, len(models) - 0.4)
    handles = [plt.Line2D([], [], ls="none", marker=_pt_style(cfg_fig, pt)["marker"],
                          color=_pt_style(cfg_fig, pt)["color"],
                          mfc=_pt_style(cfg_fig, pt)["color"] if pt == "EV" else "white", label=pt)
               for pt in ("EV", "ICE")]
    legend_below(fig, cfg_fig, handles, ncol=2)
    ax.set_ylabel("error floor, spacing RMSE (m)")
    ax.grid(alpha=0.3, lw=0.4, axis="y")
    return fig


def physics_floors_table(summary, cfg_fig):
    """tab_physics_floors (D041 (1)): mean and median task RMSE of both floors per model and
    powertrain, the number of tasks and of collided validation runs."""
    labels = cfg_fig.get("physics_model_labels", {})
    order = {m: i for i, m in enumerate(_physics_models(summary, cfg_fig))}
    rows = []
    for (model, pt), g in summary.groupby(["model", "powertrain"]):
        r = {"Model": labels.get(model, model), "Powertrain": pt, "_o": order[model]}
        for floor, name in (("in_sample", "in-sample"), ("cross_config", "cross-config.")):
            h = g[g["floor"] == floor]
            r[f"{name} mean (m)"] = float(h["task_mean_rmse"].iloc[0]) if len(h) else np.nan
            r[f"{name} median (m)"] = float(h["task_median_rmse"].iloc[0]) if len(h) else np.nan
        r["tasks"] = str(int(g["n_tasks"].max()))
        r["collided runs"] = str(int(g["n_collided"].sum()))
        rows.append(r)
    t = pd.DataFrame(rows).sort_values(["_o", "Powertrain"]).drop(columns="_o").reset_index(drop=True)
    nl = chr(10)
    tasks = "; ".join(f"{pt} {n}" for pt, n in t.drop_duplicates("Powertrain")[["Powertrain", "tasks"]].values)
    latex = dict(latex_df=t.drop(columns="tasks"),
                 header=["Model", "Power-" + nl + "train", "Mean", "Median", "Mean", "Median",
                         "Collided" + nl + "runs"],
                 groups=[("", 2), ("In-sample (m)", 2), ("Cross-config. (m)", 2), ("", 1)],
                 note=f"Tasks per model: {tasks}.")
    return t, latex


# ---------------------------------------------------------------- fig_l3_vehicles

def l3_vehicles_figure(plt, values, methods, cfg_fig, lengths=(10.0, 300.0),
                       level="L3_cross_powertrain"):
    """fig_l3_vehicles (D041 (5)): one row per method, spacing RMSE (log axis) of each EV;
    vehicles by marker shape and colour, the first length filled, the second open."""
    ev = _pt_style(cfg_fig, "EV")
    names = [m for m in methods if m in values]
    fig, ax = new_figure(plt, cfg_fig, "single_width_in", "l3_vehicles")
    vehicles = None
    for i, name in enumerate(names):
        v = comparison.at_lengths(values[name], [float(L) for L in lengths])
        v = v[v["level"] == level]
        vehicles = sorted(v["vehicle"].unique()) if vehicles is None else vehicles
        y = len(names) - 1 - i
        for j, veh in enumerate(vehicles):
            c = ev["palette"][j % len(ev["palette"])]
            mk = ev["vehicle_markers"][j % len(ev["vehicle_markers"])]
            yy = y + (j - (len(vehicles) - 1) / 2) * 0.22
            vals = [v.loc[(v["vehicle"] == veh) & (v["length_s"] == float(L)), "value"] for L in lengths]
            vals = [float(x.iloc[0]) if len(x) else np.nan for x in vals]
            same = all(np.isfinite(vals)) and np.allclose(vals, vals[0])   # D048: no context (e.g. M0)
            for k, val in enumerate(vals[:1] if same else vals):
                if np.isfinite(val):
                    ax.plot(val, yy, ls="none", marker=mk, ms=4, color=c,
                            mfc=c if k == 0 else "white", mec=c, zorder=3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([cfg_fig["methods"][n]["label"] for n in names][::-1])
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xscale("log")
    _log_ticks(ax, "x", (1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000))
    ax.set_xlabel("spacing RMSE (m)")
    ax.grid(alpha=0.3, lw=0.4, axis="x")
    nv = len(vehicles or [])
    handles = [plt.Line2D([], [], ls="none", marker=ev["vehicle_markers"][j % len(ev["vehicle_markers"])],
                          color=ev["palette"][j % len(ev["palette"])], label=veh)
               for j, veh in enumerate(vehicles or [])]
    handles += [plt.Line2D([], [], ls="none", marker="o", color="0.3", mfc="0.3" if k == 0 else "white",
                           label=f"{int(L)} s") for k, L in enumerate(lengths)]
    legend_below(fig, cfg_fig, handles, ncol=3 if nv <= 3 else 4)
    return fig


# ---------------------------------------------------------------- tables of D040 and D041

def pretrain_diagnostics_table(diags):
    """tab_pretrain_diagnostics (D040 (3) and its NOTE): per network and context length the
    closed-loop spacing RMSE on the synthetic validation set with the predicted parameters
    (M4-C: its policy), the true and the population parameters. diags: {name: DataFrame}."""
    rows = []
    for name, d in diags.items():
        for _, r in d.sort_values("length_s").iterrows():
            rows.append({"Network": name, "observed driving (s)": str(int(r["length_s"])),
                         "episodes": int(r["n"]), "predicted (m)": float(r["rmse_pred"]),
                         "true parameters (m)": float(r["rmse_true"]), "population (m)": float(r["rmse_pop"])})
    t = pd.DataFrame(rows)
    blocks = [(n, int((t["Network"] == n).sum())) for n in diags]
    latex = dict(latex_df=t.drop(columns=["Network", "episodes"]), row_groups=blocks,
                 header=["Observed driving (s)", "Predicted", "True parameters", "Population"],
                 groups=[("", 1), ("Spacing RMSE (m)", 3)])
    return t, latex


def compute_table(states, cfg_fig):
    """tab_compute (D040 (5) and its NOTE): per GPU-trained method the fitting time per fold
    (minutes) and the selected step S over folds (median and range), the number of folds and
    the devices (for the note). states: {method short name: fold_states DataFrame}."""
    rows, devices = [], set()
    for name, d in states.items():
        res = "res_train_s" in d
        t = d["res_train_s" if res else "finetune_s"].astype(float) / 60.0
        s = d["res_best_step" if res else "best_step"].astype(float)
        dev = d["res_gpu" if res else "gpu_name"]
        rows.append({"Method": cfg_fig["methods"][name]["label"], "folds": str(len(d)),
                     "time/fold median (min)": float(t.median()),
                     "time/fold range (min)": f"{t.min():.0f}-{t.max():.0f}",
                     "S median": str(int(s.median())), "S range": f"{int(s.min())}-{int(s.max())}"})
        devices.update(map(str, dev))
    t = pd.DataFrame(rows)
    folds = "; ".join(f"{n} {m}" for m, n in zip(t["Method"], t["folds"]))
    latex = dict(latex_df=t.drop(columns="folds"), header=["Method", "Median", "Range", "Median", "Range"],
                 groups=[("", 1), ("Time per fold (min)", 2), ("Selected steps $S$", 2)])
    return t, sorted(devices), latex, "Folds: " + folds + "."


def collisions_table(coll, cfg_fig, method="M4C", levels=("L2_new_vehicle", "L2_no_ice", "L3_cross_powertrain")):
    """tab_collisions (D041 (3)): collided / runs and share per level and run type for one
    method; also returns the collided runs of all other methods and the vehicles that have
    stop runs (for the note)."""
    m = coll[coll["method"] == method]
    rows = []
    for lvl in levels:
        r = {"Level": level_label(cfg_fig, lvl)}
        for rt, name in (("stop", "stop runs"), ("no_stop", "other runs")):
            g = m[(m["level"] == lvl) & (m["run_type"] == rt)]
            n, c = int(g["n_runs"].sum()), int(g["n_collided"].sum())
            r[f"{name}: collided/simulations"] = f"{c}/{n}"
            r[f"{name} (%)"] = 100.0 * c / n if n else np.nan
        rows.append(r)
    others = int(coll.loc[coll["method"] != method, "n_collided"].sum())
    stop_vehicles = sorted(coll.loc[(coll["run_type"] == "stop") & (coll["n_runs"] > 0), "vehicle"].unique())
    nl = chr(10)
    latex = dict(header=["Level", "Collided /" + nl + "simulations", chr(92) + "%", "Collided /" + nl + "simulations",
                         chr(92) + "%"],
                 groups=[("", 1), ("Stop runs", 2), ("Other runs", 2)], float_fmt="{:.1f}")
    return pd.DataFrame(rows), others, stop_vehicles, latex


def all_methods_table(values, coll, cfg_fig, methods, levels, lengths):
    """tab_all_methods (D041 (6), D045): status, vehicle-level mean spacing RMSE per level and
    length, collisions; returns (table, bold mask with the lowest value per column, LaTeX
    header, LaTeX group header)."""
    rows = []
    for name in methods:
        if name not in values:
            continue
        st = cfg_fig["methods"][name]
        r = {"Method": st["label"], "status": STATUS_SHORT.get(st["design"], st["design"])}
        v = comparison.at_lengths(values[name], [float(L) for L in lengths])
        for lvl in levels:
            for L in lengths:
                x = v.loc[(v["level"] == lvl) & (v["length_s"] == float(L)), "value"].dropna()
                r[f"{LEVEL_SHORT.get(lvl, lvl)} {int(L)} s"] = float(x.mean()) if len(x) else np.nan
        r["collisions"] = str(int(coll.loc[coll["method"] == name, "n_collided"].sum()))
        rows.append(r)
    t = pd.DataFrame(rows)
    bold = pd.DataFrame(False, index=t.index, columns=t.columns)
    for c in t.columns:
        if pd.api.types.is_numeric_dtype(t[c]) and t[c].notna().any():
            bold[c] = t[c] == t[c].min()
    header = ["Method", "status"] + [f"{int(L)} s" for _ in levels for L in lengths] + ["collisions"]
    groups = ([("", 2)] + [(level_label(cfg_fig, l).replace(", ", "," + chr(10)), len(lengths)) for l in levels]
              + [("", 1)])
    header = ["Method", "Status"] + header[2:-1] + ["Coll."]
    return t, dict(bold=bold, header=header, groups=groups, wide=True, size="footnotesize", colsep=3.5)


# ---------------------------------------------------------------- group 2 (D040 (4), D041 (2) (4), D050, D052)

def gbm_importance_table(imp, cfg_fig, n_folds):
    """tab_gbm_importance (D052): importance of each context feature (rows, in the blocks of
    config gbm_feature_blocks) for each IDM parameter (columns), mean over the n_folds L2
    folds; values that print as 0.10 or more in bold. Returns (table with the unrounded
    values, LaTeX options)."""
    labels = cfg_fig.get("gbm_feature_labels", {})
    blocks = [(str(b), list(f)) for b, f in cfg_fig["gbm_feature_blocks"]]
    order = [f for _, fs in blocks for f in fs]
    if sorted(order) != sorted(imp.index) or len(set(order)) != len(order):
        raise ValueError("gbm_feature_blocks must list every feature of the importance table exactly once")
    params = [c for c in ("v0", "T", "s0", "a", "b") if c in imp.columns]
    if len(params) != len(imp.columns):
        raise ValueError(f"unexpected importance columns: {list(imp.columns)}")
    rows = []
    for b, fs in blocks:
        for f in fs:
            rows.append({"block": b, "feature": labels.get(f, f), **{k: float(imp.loc[f, k]) for k in params}})
    t = pd.DataFrame(rows)
    ldf = t.drop(columns="block").rename(columns={"feature": "Feature"})
    bold = pd.DataFrame(False, index=ldf.index, columns=ldf.columns)
    for k in params:
        bold[k] = [float(f"{v:.2f}") >= 0.10 for v in ldf[k]]          # as printed (D052)
    math = {"v0": "$v_0$", "T": "$T$", "s0": "$s_0$", "a": "$a$", "b": "$b$"}
    note = (f"Impurity-based importance, mean over the {n_folds} folds; each column sums to 1 "
            f"before rounding; values of 0.10 or more in bold.")
    latex = dict(latex_df=ldf, bold=bold, header=["Feature"] + [math[k] for k in params],
                 row_groups=[(b, len(fs)) for b, fs in blocks], note=note)
    return t, latex


def collision_figure(plt, ex, cfg_fig):
    """fig_collision_example: gap against time of one re-simulated collision of the
    direct-acceleration network, with the observed gap and the population prior."""
    fig, ax = new_figure(plt, cfg_fig, "single_width_in", "collision_example")
    t = ex["t"]
    net, prior = cfg_fig["methods"]["M4C"], cfg_fig["methods"]["M2"]
    h_obs, = ax.plot(t, ex["s_obs"], color="k", lw=1.0, label="observed")
    h_p, = ax.plot(t, ex["s_prior"], color=prior["color"], ls="--", lw=1.0, label=prior["label"])
    k = ex["n_valid"] - 1
    h_n, = ax.plot(t[:k + 1], ex["s_net"][:k + 1], color=net["color"], lw=1.0, label=net["label"])
    h_c, = ax.plot(t[k], ex["s_net"][k], ls="none", marker="x", ms=6, mew=1.5, color=net["color"],
                   label="collision")
    ax.axhline(0.0, color="0.5", lw=0.5, zorder=1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("gap (m)")
    ax.grid(alpha=0.3, lw=0.4)
    legend_below(fig, cfg_fig, [h_obs, h_n, h_c, h_p], ncol=2)
    return fig


def residual_map_figure(plt, rm, cfg_fig):
    """fig_residual_map: mean correction (m/s2) of the residual network over the ICE folds on
    a gap x speed-difference grid, filled contours with labelled contour lines (no colour
    bar: one axes) and the dashed region holding 90 % of the observed ICE states near the
    chosen speed."""
    st = cfg_fig["style"]
    rc = cfg_fig["residual_map"]
    fig, ax = new_figure(plt, cfg_fig, "single_width_in", "residual_map")
    R = rm["R"]
    lim = max(float(np.nanmax(np.abs(R))), 1e-6)
    levels = np.linspace(-lim, lim, 13)
    ax.contourf(rm["gaps"], rm["dvs"], R, levels=levels, cmap="RdBu_r", vmin=-lim, vmax=lim)
    lines = [x for x in levels[2:-2:2] if abs(x) > 1e-12]           # D051: a few labelled lines
    cs = ax.contour(rm["gaps"], rm["dvs"], R, levels=lines, colors="0.25", linewidths=0.4)
    ax.clabel(cs, fmt=lambda x: f"{x:+.2f}".replace("-", chr(8722)), fontsize=st["font_size"] - 2.5,
              inline=True, inline_spacing=2)
    ax.contour(rm.get("hx", rm["gaps"]), rm.get("hy", rm["dvs"]), rm["H"], levels=[rm["level"]],
               colors="k", linestyles="--", linewidths=0.9)
    ax.axhline(0.0, color="0.5", lw=0.4)
    ax.set_xlabel("gap (m)")
    ax.set_ylabel("speed difference, leader $-$ follower (m/s)")
    h = [plt.Line2D([], [], color="k", ls="--", lw=0.9,
                    label=f"{int(round(100 * float(rc['mass'])))}% of observed ICE states, speed "
                          f"{rm['v_med']:.1f} {chr(177)} {float(rc['speed_band_ms']):g} m/s")]
    from matplotlib.patches import Patch
    import matplotlib
    cmap = matplotlib.colormaps["RdBu_r"]
    h = [Patch(color=cmap(0.85), label="correction > 0 (more acceleration)"),
         Patch(color=cmap(0.15), label="correction < 0 (more braking)")] + h
    legend_below(fig, cfg_fig, h, ncol=1)
    return fig
