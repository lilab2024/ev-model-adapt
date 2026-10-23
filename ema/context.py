"""Context and query runs inside a held-out task.

For each task and draw, the configurations of the task are split into query
configurations (simulated in closed loop to measure the error) and context
configurations (from which the context given to the model is cut). Contexts of
different lengths are nested: a longer context starts with the shorter one.
The split depends only on the task and the draw, never on the fold or the model.
"""
import hashlib

import numpy as np
import pandas as pd

from ema import config


def _seed(seed, *parts):
    digest = hashlib.md5("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:8]
    return [int(seed), int(digest, 16)]


def split_task(rt, task, draw, cfg=None):
    """One draw for one task.

    Returns dict(task, draw, query, pool, contexts) where query and pool are run_id lists
    and contexts maps each length (s) to a list of (run_id, t0, t1) segments, or to None
    if the context configurations are too short for that length.
    """
    cfg = cfg or config.load_yaml("evaluation.yaml")
    c = cfg["context"]
    sub = rt[rt["task"] == task]
    configs = sorted(sub["config_id"].unique())
    if len(configs) < 2:
        raise ValueError(f"Task {task} has fewer than 2 configurations")
    rng = np.random.default_rng(_seed(c["seed"], task, draw))
    order = [configs[i] for i in rng.permutation(len(configs))]
    n_query = min(max(1, int(round(c["query_fraction"] * len(configs)))), len(configs) - 1)
    query_cfg, pool_cfg = set(order[:n_query]), order[n_query:]

    query = sorted(sub.index[sub["config_id"].isin(query_cfg)])
    pool = [r for cid in pool_cfg for r in sorted(sub.index[sub["config_id"] == cid])]
    pool = [pool[i] for i in rng.permutation(len(pool))]

    dur = sub["duration_s"].astype(float)
    first_free = max(dur[pool[0]] - min(c["lengths_s"]), 0.0)
    start = round(float(rng.uniform(0.0, first_free)), 1)     # random start in the first run
    available = float(dur[pool].sum()) - start

    contexts = {}
    for length in sorted(c["lengths_s"]):
        if length > available + 1e-9:
            contexts[length] = None
            continue
        segs, remaining = [], float(length)
        for i, rid in enumerate(pool):
            t0 = start if i == 0 else 0.0
            t1 = min(dur[rid], t0 + remaining)
            if t1 > t0:
                segs.append((rid, round(t0, 1), round(t1, 1)))
                remaining -= t1 - t0
            if remaining <= 1e-9:
                break
        contexts[length] = segs
    return dict(task=task, draw=draw, query=query, pool=pool, contexts=contexts)


def context_table(rt, cfg=None):
    """Long table of all tasks and draws: task, draw, role (query / context), length_s,
    run_id, t0, t1. Query rows cover whole runs; unavailable context lengths have no rows."""
    cfg = cfg or config.load_yaml("evaluation.yaml")
    rows = []
    for task in sorted(rt["task"].unique()):
        for draw in range(cfg["context"]["draws"]):
            s = split_task(rt, task, draw, cfg)
            for rid in s["query"]:
                rows.append(dict(task=task, draw=draw, role="query", length_s=np.nan,
                                 run_id=rid, t0=0.0, t1=float(rt.loc[rid, "duration_s"])))
            for length, segs in s["contexts"].items():
                for rid, t0, t1 in segs or []:
                    rows.append(dict(task=task, draw=draw, role="context",
                                     length_s=float(length), run_id=rid, t0=t0, t1=t1))
    return pd.DataFrame(rows)
