"""Training orchestration: validate data, fit, checkpoint, and evaluate."""

import os
import random
from datetime import datetime

import numpy as np

# Set this before importing TensorFlow in this entrypoint.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf

from nowcasting.config import (
    FORECAST_LEAD_MINUTES, INPUT_CHANNELS, INPUT_LENGTH,
    NOMINAL_CADENCE_MINUTES, OUTPUT_LENGTH, RANDOM_SEED, TARGET_CHANNELS,
    TARGET_SHAPE,
)
from nowcasting.data.model_data import get_sorted_files
from nowcasting.data.provenance import ProvenanceMismatchError
from nowcasting.data.provenance import sha256_file as _file_sha256
from nowcasting.data.provenance import write_json_atomic as _write_json_atomic
from nowcasting.data.sequences import (
    _report_split_overlap, create_day_independent_splits,
)
from nowcasting.models.model_tf2 import build_model
from nowcasting.paths import MODEL_SAVE_DIR as MODEL_SAVE_DIR_PATH
from nowcasting.paths import PROCESSED_DATA_DIR
from nowcasting.training.checkpoints import (
    MODEL_SCHEMA_VERSION, checkpoint_metadata, checkpoint_metadata_path, load_compatible_weights,
    save_versioned_weights, training_backup_directory,
)
from nowcasting.training.dataset import create_dataset
from nowcasting.training.evaluation import evaluate_test_set
from nowcasting.training.losses import CoverageWeightedLoss
from nowcasting.training.metrics import build_metrics
from nowcasting.training.settings import EPOCHS, GRAD_CLIP_NORM, LEARNING_RATE

DATA_DIR = str(PROCESSED_DATA_DIR)
MODEL_SAVE_DIR = str(MODEL_SAVE_DIR_PATH)


def set_global_determinism(seed=RANDOM_SEED):
    """Seed Python, NumPy, and TensorFlow before constructing the model."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def train(
    epochs=EPOCHS,
    run_tag="",
    max_steps_per_epoch=None,
    max_validation_steps=None,
    resume_weights_path=None,
    allow_unversioned_weights=False,
):
    if epochs < 1:
        raise ValueError("Training epochs must be positive.")
    for limit in (max_steps_per_epoch, max_validation_steps):
        if limit is not None and limit < 1:
            raise ValueError("Training and validation step limits must be positive.")
    try:
        all_files = get_sorted_files(DATA_DIR)
    except (OSError, ValueError, ProvenanceMismatchError) as exc:
        print(f"Invalid model data: {exc}")
        return False
    print(f"Total validated files found: {len(all_files)}")
    if not all_files:
        print("No files found. Exiting.")
        return False

    try:
        train_sequences, val_sequences, test_sequences, split_days = (
            create_day_independent_splits(all_files, INPUT_LENGTH, OUTPUT_LENGTH)
        )
    except ValueError as exc:
        print(f"Cannot create independent split: {exc}")
        return False
    print(
        "Day-independent split: "
        f"train={len(train_sequences)}, val={len(val_sequences)}, "
        f"test={len(test_sequences)}"
    )
    print(
        f"Split days: train={split_days['train']}, "
        f"validation={split_days['validation']}, test={split_days['test']}"
    )

    if not train_sequences:
        print("No valid training sequences found. Exiting.")
        return False
    if not val_sequences:
        print("No valid validation sequences found in val split. Exiting.")
        return False
    if not test_sequences:
        print("No valid test sequences found in test split. Exiting.")
        return False

    _report_split_overlap(train_sequences, val_sequences, test_sequences)

    set_global_determinism()
    input_shape = (INPUT_LENGTH, *TARGET_SHAPE, INPUT_CHANNELS)
    model = build_model(input_shape, INPUT_LENGTH, OUTPUT_LENGTH)
    if resume_weights_path:
        load_compatible_weights(
            model, resume_weights_path, allow_unversioned=allow_unversioned_weights
        )
        print(f"Loaded compatible warm-start weights from: {resume_weights_path}")
    model.compile(
        # FIX 6: clip gradients so a rare intense-echo batch cannot blow up
        # the weights now that the heavy classes carry large loss weights.
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE, clipnorm=GRAD_CLIP_NORM
        ),
        # mean_with_sample_weight keeps loss magnitudes comparable when radar
        # coverage differs between batches instead of dividing by masked cells.
        loss=CoverageWeightedLoss(),
        metrics=[],
        weighted_metrics=build_metrics(include_per_lead=False),
    )

    model.summary()

    train_dataset = create_dataset(
        train_sequences, training=True, max_batches=max_steps_per_epoch
    )
    val_dataset = create_dataset(val_sequences, max_batches=max_validation_steps)
    test_dataset = create_dataset(test_sequences, training=False)

    steps_per_epoch = int(train_dataset.cardinality())
    validation_steps = int(val_dataset.cardinality())

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_prefix = f"{run_tag}_" if run_tag else ""
    log_filename = f"training_log_{tag_prefix}{run_timestamp}.csv"
    plot_filename = f"training_plot_{tag_prefix}{run_timestamp}.png"
    checkpoint_filename = f"best_model_{tag_prefix}{run_timestamp}.weights.h5"
    checkpoint_path = os.path.join(MODEL_SAVE_DIR, checkpoint_filename)
    backup_dir = str(training_backup_directory(model, run_tag=run_tag))

    callbacks = [
        tf.keras.callbacks.CSVLogger(os.path.join(MODEL_SAVE_DIR, log_filename)),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        # Restores optimizer, epoch, and model state after an interrupted job.
        tf.keras.callbacks.BackupAndRestore(backup_dir=backup_dir),
    ]

    print(
        f"Training config: epochs={epochs}, "
        f"steps_per_epoch={steps_per_epoch}, validation_steps={validation_steps}"
    )
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1,
    )
    if os.path.exists(checkpoint_path):
        # Evaluate and publish the actual minimum-validation-loss checkpoint,
        # even when training reached the final epoch without early stopping.
        model.load_weights(checkpoint_path)
        best_metadata = checkpoint_metadata(model)
        best_metadata["weights_sha256"] = _file_sha256(checkpoint_path)
        _write_json_atomic(
            checkpoint_metadata_path(checkpoint_path), best_metadata
        )
        print(f"Restored best validation checkpoint: {checkpoint_path}")

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.plot(history.history["loss"], label="Train Loss", marker="o")
        plt.plot(history.history["val_loss"], label="Val Loss", marker="s")
        plt.title("Training History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_SAVE_DIR, plot_filename), dpi=150)
        plt.close()
        print(f"Training plot saved: {plot_filename}")
    except Exception as exc:
        print(f"Could not save training plot: {exc}")

    final_weights_path = os.path.join(MODEL_SAVE_DIR, "final_model.weights.h5")
    save_versioned_weights(model, final_weights_path)
    print(f"Model weights saved to {final_weights_path}")

    _write_json_atomic(
        os.path.join(MODEL_SAVE_DIR, "test_sequences.json"), test_sequences
    )
    print("Test sequences saved to test_sequences.json")

    split_manifest = {
        "schema_version": 1,
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "strategy": "chronological_calendar_days_split_before_windows",
        "random_seed": RANDOM_SEED,
        "input_length": INPUT_LENGTH,
        "input_channels": INPUT_CHANNELS,
        "output_length": OUTPUT_LENGTH,
        "target_channels": TARGET_CHANNELS,
        "nominal_cadence_minutes": NOMINAL_CADENCE_MINUTES,
        "forecast_lead_minutes": list(FORECAST_LEAD_MINUTES),
        "days": split_days,
        "sequence_counts": {
            "train": len(train_sequences),
            "validation": len(val_sequences),
            "test": len(test_sequences),
        },
    }
    _write_json_atomic(
        os.path.join(MODEL_SAVE_DIR, "split_manifest.json"), split_manifest
    )

    print("Evaluating best restored weights on the untouched test split...")
    evaluation = evaluate_test_set(model, test_dataset)
    evaluation.update(
        {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "weights_path": final_weights_path,
            "weights_sha256": _file_sha256(final_weights_path),
            "checkpoint_metadata": checkpoint_metadata(model),
            "split_manifest": split_manifest,
        }
    )
    run_results_path = os.path.join(
        MODEL_SAVE_DIR, f"test_results_{tag_prefix}{run_timestamp}.json"
    )
    _write_json_atomic(run_results_path, evaluation)
    _write_json_atomic(os.path.join(MODEL_SAVE_DIR, "test_results.json"), evaluation)
    print(f"Held-out model and persistence results saved to {run_results_path}")
    return True


def main():
    return 0 if train(epochs=EPOCHS, run_tag="") else 1


if __name__ == "__main__":
    raise SystemExit(main())
