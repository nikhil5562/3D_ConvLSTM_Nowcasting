from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from nowcasting.config import FORECAST_HORIZON_MINUTES
from nowcasting.config import FORECAST_LEAD_MINUTES
from nowcasting.config import INPUT_CHANNELS
from nowcasting.config import OUTPUT_LENGTH
from nowcasting.config import TARGET_SHAPE
from nowcasting.models.model_tf2 import build_model
from nowcasting.training.checkpoints import CheckpointCompatibilityError
from nowcasting.training.metrics import ContingencyMetric
from nowcasting.training.losses import CoverageWeightedLoss
from nowcasting.data.sequences import _report_split_overlap
from nowcasting.training.checkpoints import _architecture_sha256
from nowcasting.training.checkpoints import checkpoint_metadata_path
from nowcasting.data.sequences import create_day_independent_splits
from nowcasting.training.evaluation import evaluate_test_set
from nowcasting.training.checkpoints import load_compatible_weights
from nowcasting.training.checkpoints import save_versioned_weights
from nowcasting.data.sequences import sequence_generator
from nowcasting.training.checkpoints import training_backup_directory
from nowcasting.training.losses import weighted_loss


def test_default_horizon_is_eight_15_minute_leads_to_120_minutes():
    assert OUTPUT_LENGTH == 8
    assert FORECAST_LEAD_MINUTES == (15, 30, 45, 60, 75, 90, 105, 120)
    assert FORECAST_HORIZON_MINUTES == 120


def test_model_build_respects_shared_input_and_forecast_contract():
    model = build_model(
        (10, *TARGET_SHAPE, INPUT_CHANNELS), input_length=10, output_length=8
    )

    assert model.input_shape == (None, 10, 16, 120, 120, 2)
    assert model.output_shape == (None, 8, 16, 120, 120, 1)


def test_stateful_csi_accumulates_counts_instead_of_averaging_batches():
    metric = ContingencyMetric("csi", 20.0)
    hit = tf.constant([[[[[[0.5]]]]]], dtype=tf.float32)
    metric.update_state(hit, hit)

    observations = tf.ones((1, 1, 1, 10, 10, 1), dtype=tf.float32) * 0.5
    misses = tf.zeros_like(observations)
    metric.update_state(observations, misses)

    assert float(metric.result()) == pytest.approx(1.0 / 101.0)


def test_per_lead_metric_ignores_cells_without_radar_coverage():
    metric = ContingencyMetric("pod", 20.0, lead_index=1)
    truth = tf.ones((1, 2, 1, 1, 2, 1), dtype=tf.float32) * 0.5
    prediction = tf.constant(
        [[[[[[0.0], [0.0]]]], [[[[0.5], [0.0]]]]]], dtype=tf.float32
    )
    coverage = tf.constant(
        [[[[[[1.0], [1.0]]]], [[[[1.0], [0.0]]]]]], dtype=tf.float32
    )

    metric.update_state(truth, prediction, sample_weight=coverage)

    assert float(metric.hits) == pytest.approx(1.0)
    assert float(metric.misses) == pytest.approx(0.0)
    assert float(metric.result()) == pytest.approx(1.0)


def test_coverage_mask_removes_masked_targets_from_weighted_loss():
    prediction = tf.zeros((1, 1, 1, 1, 2, 1), dtype=tf.float32)
    truth_a = tf.constant([[[[[[0.5], [0.0]]]]]], dtype=tf.float32)
    truth_b = tf.constant([[[[[[0.5], [1.0]]]]]], dtype=tf.float32)
    coverage = tf.constant([[[[[[1.0], [0.0]]]]]], dtype=tf.float32)

    loss_a = tf.reduce_sum(weighted_loss(truth_a, prediction) * coverage)
    loss_b = tf.reduce_sum(weighted_loss(truth_b, prediction) * coverage)

    assert float(loss_a) == pytest.approx(float(loss_b))


def test_coverage_weighted_loss_normalizes_by_observed_weight():
    prediction = tf.zeros((1, 1, 1, 1, 3, 1), dtype=tf.float32)
    truth = tf.constant([[[[[[0.5], [0.5], [1.0]]]]]], dtype=tf.float32)
    coverage = tf.constant([[[[[[1.0], [0.5], [0.0]]]]]], dtype=tf.float32)

    loss_field = weighted_loss(truth, prediction)
    expected = tf.reduce_sum(loss_field * coverage) / tf.reduce_sum(coverage)
    actual = CoverageWeightedLoss()(truth, prediction, sample_weight=coverage)

    assert float(actual) == pytest.approx(float(expected))


def _file_records_for_day(day, count=6):
    start = datetime.strptime(day, "%Y-%m-%d")
    records = []
    for index in range(count):
        timestamp = start + timedelta(minutes=15 * index)
        records.append(
            {
                "path": str(
                    Path("processed_data")
                    / f"RCTLS_{timestamp:%d%b%Y_%H%M%S}_L2B_STD.npy"
                ),
                "time": timestamp,
                "echo": 1.0,
            }
        )
    return records


def test_day_split_happens_before_windows_and_has_no_day_overlap():
    files = []
    for day in ("2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"):
        files.extend(_file_records_for_day(day))

    train, validation, test, days = create_day_independent_splits(
        files, input_len=2, output_len=1, train_ratio=0.5, val_ratio=0.25
    )

    assert days["train"] == ["2025-01-01", "2025-01-02"]
    assert days["validation"] == ["2025-01-03"]
    assert days["test"] == ["2025-01-04"]
    assert train and validation and test
    assert all(len(sequence) == 3 for sequence in train + validation + test)
    _report_split_overlap(train, validation, test)


def test_generator_adds_input_coverage_channel_and_target_weights(make_model_frame):
    paths = [
        make_model_frame(index, dbz=value, coverage=0.25 + 0.5 * index)
        for index, value in enumerate((10.0, 20.0))
    ]

    x, y, target_coverage = next(
        sequence_generator([paths], input_len=1, output_len=1)
    )

    assert x.shape[-1] == INPUT_CHANNELS
    assert x[0, 0, 0, 0, 0] == pytest.approx(10.0 / 80.0)
    assert x[0, 0, 0, 0, 1] == pytest.approx(0.25)
    assert y[0, 0, 0, 0, 0] == pytest.approx(20.0 / 80.0)
    assert target_coverage[0, 0, 0, 0, 0] == pytest.approx(0.75)


def _tiny_model(units=1):
    inputs = tf.keras.Input(shape=(2,), name="checkpoint_input")
    outputs = tf.keras.layers.Dense(units, name="checkpoint_output")(inputs)
    return tf.keras.Model(inputs, outputs, name=f"checkpoint_model_{units}")


def test_versioned_checkpoint_loads_only_matching_architecture(tmp_path):
    weights_path = tmp_path / "tiny.weights.h5"
    original = _tiny_model(1)
    save_versioned_weights(original, weights_path)

    assert checkpoint_metadata_path(weights_path).exists()
    load_compatible_weights(_tiny_model(1), weights_path)

    with pytest.raises(CheckpointCompatibilityError, match="incompatible"):
        load_compatible_weights(_tiny_model(2), weights_path)


def test_unversioned_checkpoint_is_rejected(tmp_path):
    weights_path = tmp_path / "legacy.weights.h5"
    _tiny_model(1).save_weights(weights_path)

    with pytest.raises(CheckpointCompatibilityError, match="metadata is missing"):
        load_compatible_weights(_tiny_model(1), weights_path)


def test_checkpoint_signature_ignores_keras_auto_name_counters():
    first = tf.keras.Sequential([tf.keras.Input((2,)), tf.keras.layers.Dense(1)])
    second = tf.keras.Sequential([tf.keras.Input((2,)), tf.keras.layers.Dense(1)])

    assert _architecture_sha256(first) == _architecture_sha256(second)


def test_training_backup_is_namespaced_by_architecture():
    one_output = training_backup_directory(_tiny_model(1), run_tag="test")
    two_outputs = training_backup_directory(_tiny_model(2), run_tag="test")

    assert one_output != two_outputs
    assert f"v1_" in str(one_output)


def test_held_out_evaluation_reports_each_lead_and_persistence():
    inputs = tf.keras.Input(shape=(10, 1, 1, 2, INPUT_CHANNELS))
    outputs = tf.keras.layers.Lambda(
        lambda value: tf.repeat(value[:, -1:, ..., :1], OUTPUT_LENGTH, axis=1)
    )(inputs)
    model = tf.keras.Model(inputs, outputs)

    x = np.zeros((1, 10, 1, 1, 2, INPUT_CHANNELS), dtype=np.float32)
    x[..., 1] = 1.0
    x[:, -1, ..., 0] = 0.5
    y = np.full((1, OUTPUT_LENGTH, 1, 1, 2, 1), 0.5, dtype=np.float32)
    coverage = np.ones_like(y)

    results = evaluate_test_set(model, [(x, y, coverage)])

    assert results["model"]["coverage_weighted_loss"] == pytest.approx(0.0)
    assert results["persistence"]["coverage_weighted_loss"] == pytest.approx(0.0)
    assert results["model"]["overall"]["20_dbz"]["csi"] == pytest.approx(1.0)
    assert len(results["model"]["per_lead"]) == OUTPUT_LENGTH
    assert results["model"]["per_lead"][-1]["lead_minutes"] == 120
