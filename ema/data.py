"""Read the shared processed data (read-only); the md5 checksum is verified first."""
import hashlib

import pandas as pd

from ema import config


def file_md5(path, chunk_size=1 << 20):
    """md5 hex digest of a file, read in chunks."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def check_md5(path, expected):
    """Raise ValueError if the md5 of `path` differs from `expected`."""
    actual = file_md5(path)
    if actual != expected:
        raise ValueError(f"md5 mismatch for {path}: expected {expected}, got {actual}")
    return actual


def load_manifest():
    """One row per run (including excluded runs)."""
    return pd.read_csv(config.processed_paths()["manifest"])


def load_qc():
    """Quality-check metrics per processed run, with the qc_pass flag."""
    return pd.read_csv(config.processed_paths()["qc"])


def load_runs():
    """Manifest joined with QC metrics; excluded runs (not processed) have NaN QC values."""
    return load_manifest().merge(load_qc(), on="run_id", how="left", validate="one_to_one")


def load_trajectories(columns=None):
    """10 Hz trajectories of all processed runs.

    The md5 checksum and the expected row counts in config/data.yaml are checked;
    any mismatch raises ValueError.
    """
    cfg = config.load_yaml("data.yaml")["processed"]
    path = config.processed_paths()["trajectories"]
    check_md5(path, cfg["md5"]["trajectories"])
    if columns is not None and "run_id" not in columns:
        columns = ["run_id", *columns]
    traj = pd.read_parquet(path, columns=columns)

    expected = cfg["expected_rows"]
    if len(traj) != expected["total"]:
        raise ValueError(f"Expected {expected['total']} rows, found {len(traj)}")
    ev_ids = set(load_manifest().query("powertrain == 'EV'")["run_id"])
    n_ev = int(traj["run_id"].isin(ev_ids).sum())
    if n_ev != expected["ev"]:
        raise ValueError(f"Expected {expected['ev']} EV rows, found {n_ev}")
    return traj
