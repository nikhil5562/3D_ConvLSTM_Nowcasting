"""
Retrain entrypoint that uses the leakage-safe training pipeline in train.py.
"""

import os
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="ignore")

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nowcasting.paths import MODEL_SAVE_DIR
from nowcasting.training.train import train

RETRAIN_EPOCHS = int(os.environ.get("RETRAIN_EPOCHS", "20"))
MAX_STEPS = os.environ.get("RETRAIN_MAX_STEPS")
MAX_VAL_STEPS = os.environ.get("RETRAIN_MAX_VAL_STEPS")
# A changed architecture must start fresh by default. Set RETRAIN_RESUME=1
# only for a versioned, compatible warm start.
RESUME = os.environ.get("RETRAIN_RESUME", "0") == "1"
ALLOW_UNVERSIONED = os.environ.get("RETRAIN_ALLOW_UNVERSIONED", "0") == "1"
RESUME_WEIGHTS = os.environ.get(
    "RETRAIN_WEIGHTS", str(MODEL_SAVE_DIR / "final_model.weights.h5")
)


def main():
    print("=" * 60)
    print("RETRAINING MODEL")
    print("=" * 60)
    success = train(
        epochs=RETRAIN_EPOCHS,
        run_tag="retrain",
        max_steps_per_epoch=int(MAX_STEPS) if MAX_STEPS else None,
        max_validation_steps=int(MAX_VAL_STEPS) if MAX_VAL_STEPS else None,
        resume_weights_path=RESUME_WEIGHTS if RESUME else None,
        allow_unversioned_weights=ALLOW_UNVERSIONED,
    )
    if success:
        print("\nTraining complete.")
        return 0
    else:
        print("\nTraining did not complete.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
