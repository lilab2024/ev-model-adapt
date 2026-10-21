import hashlib
import os

import pytest

from ema import config, data


def test_file_md5_matches_hashlib(tmp_path):
    content = b"abc" * 100_000
    p = tmp_path / "x.bin"
    p.write_bytes(content)
    assert data.file_md5(p, chunk_size=1000) == hashlib.md5(content).hexdigest()


def test_check_md5_rejects_mismatch(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    with pytest.raises(ValueError):
        data.check_md5(p, "0" * 32)


needs_data = pytest.mark.skipif(not os.environ.get(config.ENV_DATA_ROOT),
                                reason="EMA_DATA_ROOT not set")


@needs_data
def test_shared_data_md5_and_row_counts():
    traj = data.load_trajectories(columns=["t"])
    expected = config.load_yaml("data.yaml")["processed"]["expected_rows"]
    assert len(traj) == expected["total"]


@needs_data
def test_runs_table_one_row_per_run():
    runs = data.load_runs()
    assert runs["run_id"].is_unique
    assert set(runs["powertrain"]) == {"ICE", "EV"}
