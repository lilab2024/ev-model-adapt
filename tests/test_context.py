import os

import pandas as pd
import pytest

from ema import config, context, data, protocol

CFG = {
    "runs": {"qc_pass_only": True, "include_stop_runs": True},
    "task_keys": ["powertrain", "vehicle", "gap"],
    "config_keys": ["vehicle", "gap", "profile", "desired_mph", "v_free_mph", "v_reduced_mph"],
    "context": {"query_fraction": 0.5, "lengths_s": [10, 30, 60], "draws": 3, "seed": 42},
}


def synthetic():
    """Tasks S and L: 4 configurations x 2 runs of 25 s. Task T: 2 configurations x 1 run."""
    rows = []
    for gap, n_cfg, reps in [("S", 4, 2), ("L", 4, 2), ("T", 2, 1)]:
        for k in range(n_cfg):
            for rep in range(1, reps + 1):
                rows.append(dict(run_id=f"EV_A_{gap}_{k}_r{rep}", powertrain="EV", vehicle="A",
                                 gap=gap, profile="dip", desired_mph=0.0, v_free_mph=55.0,
                                 v_reduced_mph=float(10 * k), qc_pass=True, exclude=False,
                                 duration_s=25.0))
    return protocol.run_table(pd.DataFrame(rows), CFG)


def all_splits(rt):
    return [context.split_task(rt, t, d, CFG)
            for t in sorted(rt["task"].unique()) for d in range(CFG["context"]["draws"])]


def test_query_and_context_are_disjoint_and_configs_not_split():
    rt = synthetic()
    for s in all_splits(rt):
        assert s["query"] and s["pool"]
        assert not set(s["query"]) & set(s["pool"])
        assert not set(rt.loc[s["query"], "config_id"]) & set(rt.loc[s["pool"], "config_id"])


def test_context_length_is_exact():
    for s in all_splits(synthetic()):
        for length, segs in s["contexts"].items():
            if segs is not None:
                assert abs(sum(t1 - t0 for _, t0, t1 in segs) - length) < 0.11


def test_contexts_are_nested():
    for s in all_splits(synthetic()):
        avail = [(L, v) for L, v in sorted(s["contexts"].items()) if v is not None]
        for (_, short), (_, long) in zip(avail, avail[1:]):
            k = len(short) - 1
            assert short[:k] == long[:k]
            assert short[k][:2] == long[k][:2] and short[k][2] <= long[k][2]


def test_unavailable_length_is_none():
    s = context.split_task(synthetic(), "EV | A | T", 0, CFG)
    assert s["contexts"][60] is None
    assert s["contexts"][10] is not None


def test_split_is_deterministic_and_draws_differ():
    rt = synthetic()
    a = context.split_task(rt, "EV | A | S", 0, CFG)
    assert a == context.split_task(rt, "EV | A | S", 0, CFG)
    others = [context.split_task(rt, "EV | A | S", d, CFG) for d in (1, 2)]
    assert any(o["query"] != a["query"] or o["contexts"] != a["contexts"] for o in others)


needs_data = pytest.mark.skipif(not os.environ.get(config.ENV_DATA_ROOT),
                                reason="EMA_DATA_ROOT not set")


@needs_data
def test_real_data_every_task_has_query_and_shortest_context():
    cfg = config.load_yaml("evaluation.yaml")
    rt = protocol.run_table(data.load_runs(), cfg)
    shortest = min(cfg["context"]["lengths_s"])
    for task in rt["task"].unique():
        for draw in range(cfg["context"]["draws"]):
            s = context.split_task(rt, task, draw, cfg)
            assert s["query"] and s["contexts"][shortest] is not None
