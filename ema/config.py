"""Load configuration files and resolve data paths."""
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"
ENV_DATA_ROOT = "EMA_DATA_ROOT"


def load_yaml(name):
    """Read config/<name> and return its contents as a dict."""
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_root():
    """Shared data folder, taken only from the environment variable EMA_DATA_ROOT."""
    value = os.environ.get(ENV_DATA_ROOT)
    if not value:
        raise RuntimeError(f"Set the environment variable {ENV_DATA_ROOT} "
                           "to the shared data folder")
    root = Path(value)
    if not root.is_dir():
        raise FileNotFoundError(f"{ENV_DATA_ROOT} does not point to a folder: {root}")
    return root


def processed_paths():
    """Paths of the processed data files listed in config/data.yaml."""
    cfg = load_yaml("data.yaml")["processed"]
    folder = data_root() / cfg["dir"]
    return {key: folder / name for key, name in cfg["files"].items()}
