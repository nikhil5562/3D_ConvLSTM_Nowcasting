"""
Diagnostic script to verify model/data integrity and inference viability.
"""

import glob
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nowcasting.models.model_tf2 import build_model
from nowcasting.config import DBZ_MAX
from nowcasting.config import DBZ_MIN
from nowcasting.config import INPUT_CHANNELS
from nowcasting.config import INPUT_LENGTH
from nowcasting.config import OUTPUT_LENGTH
from nowcasting.config import TARGET_SHAPE
from nowcasting.paths import MODEL_SAVE_DIR
from nowcasting.paths import PROCESSED_DATA_DIR
from nowcasting.training.train import load_compatible_weights
from nowcasting.training.train import _prepare_coverage

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="ignore")

CONT_MIN_SEC = 800
CONT_MAX_SEC = 1000

WEIGHTS_PATH = str(MODEL_SAVE_DIR / "final_model.weights.h5")
DATA_DIR = str(PROCESSED_DATA_DIR)


def parse_timestamp(filename):
    basename = os.path.basename(filename)
    parts = basename.split("_")
    dt_str = parts[1] + "_" + parts[2]
    return datetime.strptime(dt_str, "%d%b%Y_%H%M%S")


def _ensure_3d_shape(data, target_shape):
    array = np.asarray(data)
    if array.ndim > 3:
        array = np.squeeze(array)
    if array.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {array.shape}")

    target_d, target_h, target_w = target_shape
    data_d, data_h, data_w = array.shape
    array = array[:target_d, :target_h, :target_w]

    pad_d = max(0, target_d - data_d)
    pad_h = max(0, target_h - data_h)
    pad_w = max(0, target_w - data_w)
    if pad_d or pad_h or pad_w:
        array = np.pad(array, ((0, pad_d), (0, pad_h), (0, pad_w)), mode="constant")
    return array


def _prepare_frame(file_path):
    data = np.load(file_path)
    data = _ensure_3d_shape(data, TARGET_SHAPE)
    data = np.nan_to_num(data, nan=0.0, posinf=DBZ_MAX, neginf=DBZ_MIN)
    data = np.clip(data, DBZ_MIN, DBZ_MAX).astype(np.float32)
    return data / DBZ_MAX


def _get_sorted_files(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
    valid = []
    for file_path in files:
        try:
            valid.append({"path": file_path, "time": parse_timestamp(file_path)})
        except Exception:
            continue
    valid.sort(key=lambda x: x["time"])
    return valid


def _find_contiguous_window(sorted_files, seq_len):
    for i in range(len(sorted_files) - seq_len + 1):
        ok = True
        for j in range(1, seq_len):
            diff = (sorted_files[i + j]["time"] - sorted_files[i + j - 1]["time"]).total_seconds()
            if not (CONT_MIN_SEC < diff < CONT_MAX_SEC):
                ok = False
                break
        if ok:
            return sorted_files[i : i + seq_len]
    return None


def main():
    status = {
        "weights_found": False,
        "model_built": False,
        "weights_loaded": False,
        "data_found": False,
        "prediction_ok": False,
        "prediction_nontrivial": False,
    }

    print("=" * 60)
    print("MODEL AND DATA DIAGNOSTIC")
    print("=" * 60)

    print("\n1. CHECKING MODEL WEIGHTS...")
    if os.path.exists(WEIGHTS_PATH):
        status["weights_found"] = True
        file_size_mb = os.path.getsize(WEIGHTS_PATH) / (1024 * 1024)
        print(f"  OK: weights found ({file_size_mb:.2f} MB)")
    else:
        print(f"  FAIL: weights not found at {WEIGHTS_PATH}")

    print("\n2. CHECKING MODEL ARCHITECTURE...")
    model = None
    try:
        input_shape = (INPUT_LENGTH, *TARGET_SHAPE, INPUT_CHANNELS)
        model = build_model(input_shape, INPUT_LENGTH, OUTPUT_LENGTH)
        status["model_built"] = True
        print(f"  OK: model built, params={model.count_params():,}")
    except Exception as exc:
        print(f"  FAIL: model build failed: {exc}")

    if model is not None and status["weights_found"]:
        try:
            load_compatible_weights(model, WEIGHTS_PATH)
            status["weights_loaded"] = True
            print("  OK: weights loaded")
        except Exception as exc:
            print(f"  FAIL: could not load weights: {exc}")

    print("\n3. CHECKING PREPROCESSED DATA...")
    sorted_files = _get_sorted_files(DATA_DIR)
    print(f"  Files found: {len(sorted_files)}")
    if sorted_files:
        status["data_found"] = True
        sample = np.load(sorted_files[0]["path"], mmap_mode="r")
        print(f"  Sample file: {os.path.basename(sorted_files[0]['path'])}")
        print(f"  Raw sample shape: {sample.shape}")
        print(f"  Raw sample dtype: {sample.dtype}")
        print(f"  Raw sample range: [{float(np.nanmin(sample)):.2f}, {float(np.nanmax(sample)):.2f}] dBZ")
    else:
        print("  FAIL: no processed .npy files found")

    print("\n4. TESTING MODEL PREDICTION...")
    if status["model_built"] and status["weights_loaded"] and status["data_found"]:
        try:
            seq_len = INPUT_LENGTH + OUTPUT_LENGTH
            window = _find_contiguous_window(sorted_files, seq_len)
            if window is None:
                raise RuntimeError(
                    f"No contiguous {seq_len}-frame window found."
                )

            input_files = [item["path"] for item in window[:INPUT_LENGTH]]
            x_frames = []
            for file_path in input_files:
                frame = _prepare_frame(file_path)
                frame = np.expand_dims(frame, axis=-1)
                x_frames.append(
                    np.concatenate([frame, _prepare_coverage(file_path)], axis=-1)
                )

            x_test = np.stack(x_frames).astype(np.float32)
            x_test = np.expand_dims(x_test, axis=0)
            print(f"  Input tensor shape: {x_test.shape}")
            print(f"  Input tensor range: [{x_test.min():.4f}, {x_test.max():.4f}]")

            y_pred = model.predict(x_test, verbose=0)
            pred_std = float(np.std(y_pred))
            pred_max = float(np.max(y_pred))
            print(f"  Prediction shape: {y_pred.shape}")
            print(f"  Prediction range: [{float(np.min(y_pred)):.4f}, {pred_max:.4f}]")
            print(f"  Prediction std: {pred_std:.6f}")

            if pred_std < 1e-4 or pred_max < 1e-4:
                print("  WARN: prediction is nearly constant/zero")
            else:
                print("  OK: prediction appears non-trivial")
                status["prediction_nontrivial"] = True
            status["prediction_ok"] = True
        except Exception as exc:
            print(f"  FAIL: prediction test failed: {exc}")
    else:
        print("  SKIP: prediction test skipped due to earlier failures")

    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    for key, value in status.items():
        print(f"{key}: {'OK' if value else 'FAIL'}")

    hard_checks_ok = all(
        [
            status["weights_found"],
            status["model_built"],
            status["weights_loaded"],
            status["data_found"],
            status["prediction_ok"],
            status["prediction_nontrivial"],
        ]
    )

    if hard_checks_ok:
        print("\nOVERALL: PASS")
        print("Next step: python scripts/predict.py")
        exit_code = 0
    else:
        print("\nOVERALL: FAIL")
        print("Fix failed checks, then rerun scripts/diagnose_model.py")
        exit_code = 1
    print("=" * 60)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
