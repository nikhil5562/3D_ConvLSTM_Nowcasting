import os
from pathlib import Path


def _path_from_env(name, default):
    return Path(os.environ.get(name, default)).expanduser()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_AUG_DIR = PROJECT_ROOT / "data_AUG"
L2B_INPUT_DIR = _path_from_env("NOWCAST_L2B_DIR", DATA_DIR / "nc")
GRIDDED_DATA_DIR = _path_from_env("NOWCAST_GRIDDED_DIR", PROJECT_ROOT / "3d_data" / "gridded")
PROCESSED_DATA_DIR = _path_from_env("NOWCAST_PROCESSED_DIR", PROJECT_ROOT / "processed_data")
MODEL_SAVE_DIR = PROJECT_ROOT / "saved_models"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"
