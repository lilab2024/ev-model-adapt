import pytest

from ema import config


def test_yaml_files_load():
    assert "processed" in config.load_yaml("data.yaml")
    assert "seed" in config.load_yaml("experiment.yaml")


def test_data_root_requires_env(monkeypatch):
    monkeypatch.delenv(config.ENV_DATA_ROOT, raising=False)
    with pytest.raises(RuntimeError):
        config.data_root()


def test_data_root_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv(config.ENV_DATA_ROOT, str(tmp_path))
    assert config.data_root() == tmp_path


def test_md5_is_32_hex_characters():
    md5 = config.load_yaml("data.yaml")["processed"]["md5"]["trajectories"]
    assert len(md5) == 32
    int(md5, 16)
