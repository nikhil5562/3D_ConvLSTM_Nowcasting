"""
Diagnostic script to verify model/data integrity and inference viability.
"""

import os
import sys
from pathlib import Path

import numpy as np

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nowcasting.models.model_tf2 import build_model
from nowcasting.config import INPUT_CHANNELS
from nowcasting.config import INPUT_LENGTH
from nowcasting.config import OUTPUT_LENGTH
from nowcasting.config import TARGET_SHAPE
from nowcasting.paths import MODEL_SAVE_DIR
from nowcasting.paths import PROCESSED_DATA_DIR
from nowcasting.training.checkpoints import load_compatible_weights
from nowcasting.data.model_data import get_sorted_files as _get_sorted_files
from nowcasting.data.model_data import load_and_normalize_sequence
from nowcasting.data.timestamps import find_contiguous_window as _find_contiguous_window

WEIGHTS_PATH = str(MODEL_SAVE_DIR / "final_model.weights.h5")
DATA_DIR = str(PROCESSED_DATA_DIR)


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
    try:
        sorted_files = _get_sorted_files(DATA_DIR)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"  FAIL: invalid model data: {exc}")
        return 1
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
            x_test = load_and_normalize_sequence(input_files, include_coverage=True)[None, ...]
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
