import importlib
import csv
import json
import warnings

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from nowcasting.data import model_data
from nowcasting.training import dataset as dataset_module
from nowcasting.training.losses import CoverageWeightedLoss


@pytest.mark.parametrize("limit, expected_batches", [(None, 2), (1, 1)])
def test_every_epoch_trains_and_validates_including_partial_batches(
    monkeypatch, make_model_frame, limit, expected_batches,
):
    shape = (1, 2, 2)
    monkeypatch.setattr(model_data, "TARGET_SHAPE", shape)
    monkeypatch.setattr(dataset_module, "TARGET_SHAPE", shape)
    monkeypatch.setattr(dataset_module, "INPUT_LENGTH", 2)
    monkeypatch.setattr(dataset_module, "OUTPUT_LENGTH", 1)
    frames = [make_model_frame(index, dbz=10.0 + index) for index in range(6)]
    sequences = [frames[index:index + 3] for index in range(4)]
    training = dataset_module.create_dataset(sequences, training=True, batch_size=3, max_batches=limit)
    validation = dataset_module.create_dataset(sequences, batch_size=3, max_batches=limit)
    assert int(training.cardinality()) == expected_batches
    assert int(validation.cardinality()) == expected_batches

    tf.keras.utils.set_random_seed(42)
    inputs = tf.keras.Input((2, *shape, 2))
    last_frame = tf.keras.layers.Lambda(lambda value: value[:, -1:, ..., :1])(inputs)
    outputs = tf.keras.layers.Dense(1, use_bias=False, kernel_initializer="ones")(last_frame)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="sgd", loss=CoverageWeightedLoss())

    class BatchCounts(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.training = []
            self.validation = []

        def on_epoch_begin(self, epoch, logs=None):
            self.training.append(0)
            self.validation.append(0)

        def on_train_batch_end(self, batch, logs=None):
            self.training[-1] += 1

        def on_test_batch_end(self, batch, logs=None):
            self.validation[-1] += 1

    counts = BatchCounts()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        history = model.fit(training, validation_data=validation, epochs=3,
                            callbacks=[counts], verbose=0)
    assert counts.training == [expected_batches] * 3
    assert counts.validation == [expected_batches] * 3
    assert int(model.optimizer.iterations) == 3 * expected_batches
    assert np.isfinite(history.history["loss"]).all()
    assert np.isfinite(history.history["val_loss"]).all()
    assert not any("ran out of data" in str(item.message) for item in captured)


def test_training_rejects_corrupt_data_before_building_model(monkeypatch, make_model_frame, tmp_path, capsys):
    training = importlib.import_module("nowcasting.training.train")
    frame = make_model_frame()
    model_data.coverage_path(frame).unlink()
    monkeypatch.setattr(training, "DATA_DIR", str(tmp_path))

    def unexpected_model(*args, **kwargs):
        pytest.fail("Model must not be built for invalid data")

    monkeypatch.setattr(training, "build_model", unexpected_model)
    assert training.train() is False
    assert "Invalid model data" in capsys.readouterr().out


def test_prediction_rejects_corrupt_data_before_building_model(monkeypatch, make_model_frame, tmp_path, capsys):
    prediction = importlib.import_module("scripts.predict")
    frames = [make_model_frame(index) for index in range(3)]
    manifest = tmp_path / "test_sequences.json"
    manifest.write_text(json.dumps([[str(path) for path in frames]]), encoding="utf-8")
    model_data.coverage_path(frames[0]).unlink()
    monkeypatch.setattr(prediction, "INPUT_LENGTH", 2)
    monkeypatch.setattr(prediction, "OUTPUT_LENGTH", 1)
    monkeypatch.setattr(prediction, "TEST_SPLIT_PATH", manifest)

    def unexpected_model(*args, **kwargs):
        pytest.fail("Model must not be built for invalid data")

    monkeypatch.setattr(prediction, "build_model", unexpected_model)
    assert prediction.main() == 1
    assert "missing" in capsys.readouterr().out


def test_training_run_writes_compatible_checkpoints_and_held_out_results(
    monkeypatch, make_model_frame, tmp_path,
):
    training = importlib.import_module("nowcasting.training.train")
    checkpoints = importlib.import_module("nowcasting.training.checkpoints")
    evaluation = importlib.import_module("nowcasting.training.evaluation")
    shape = (1, 2, 2)
    for module in (model_data, dataset_module, training, checkpoints):
        monkeypatch.setattr(module, "TARGET_SHAPE", shape)
    for module in (dataset_module, training, checkpoints):
        monkeypatch.setattr(module, "INPUT_LENGTH", 2)
        monkeypatch.setattr(module, "OUTPUT_LENGTH", 1)
    monkeypatch.setattr(evaluation, "OUTPUT_LENGTH", 1)
    for module in (training, checkpoints, evaluation):
        monkeypatch.setattr(module, "FORECAST_LEAD_MINUTES", (15,))
    monkeypatch.setattr(checkpoints, "FORECAST_HORIZON_MINUTES", 15)
    output = tmp_path / "models"
    monkeypatch.setattr(training, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(training, "MODEL_SAVE_DIR", str(output))
    monkeypatch.setattr(checkpoints, "MODEL_SAVE_DIR", output)
    for day in range(3):
        for index in range(4):
            make_model_frame(day * 96 + index)

    def tiny_model(input_shape, input_length, output_length):
        inputs = tf.keras.Input(input_shape)
        last_frame = tf.keras.layers.Lambda(lambda value: value[:, -1:, ..., :1])(inputs)
        outputs = tf.keras.layers.Dense(1, use_bias=False, kernel_initializer="ones")(last_frame)
        model = tf.keras.Model(inputs, outputs)
        return model

    monkeypatch.setattr(training, "build_model", tiny_model)
    assert training.train(epochs=2, max_steps_per_epoch=1, max_validation_steps=1)
    # Loading the best checkpoint can restore an earlier optimizer iteration.
    # The CSV records the completed epochs before that restoration.
    with next(output.glob("training_log_*.csv")).open(newline="") as handle:
        epochs = list(csv.DictReader(handle))
    assert [row["epoch"] for row in epochs] == ["0", "1"]
    assert all(np.isfinite(float(row["val_loss"])) for row in epochs)
    results = json.loads((output / "test_results.json").read_text(encoding="utf-8"))
    assert results["model"]["evaluated_coverage_weight"] > 0
    assert len(results["model"]["per_lead"]) == 1
    assert results["split_manifest"]["sequence_counts"]["test"] == 2
    assert (output / "test_sequences.json").is_file()
    assert (output / "split_manifest.json").is_file()
    assert len(list(output.glob("best_model_*.weights.h5"))) == 1
    checkpoints.load_compatible_weights(
        tiny_model((2, *shape, 2), 2, 1), output / "final_model.weights.h5",
    )
