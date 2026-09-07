import json

import numpy as np
import pytest

from nowcasting.data.model_data import (
    coverage_path, get_sorted_files, load_and_normalize_sequence, load_frame,
)
from nowcasting.data.provenance import (
    ProvenanceMismatchError, manifest_path, write_artifact_manifest,
)
from nowcasting.data.sequences import sequence_generator


def change_manifest(frame, change):
    path = manifest_path(frame)
    payload = json.loads(path.read_text(encoding="utf-8"))
    change(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_training_and_inference_have_identical_inputs(make_model_frame):
    first = make_model_frame(0, dbz=10.0, coverage=0.25)
    second = make_model_frame(1, dbz=20.0, coverage=0.75)
    x, y, weights = next(sequence_generator([[first, second]], 1, 1))
    np.testing.assert_array_equal(x, load_and_normalize_sequence([first], include_coverage=True))
    np.testing.assert_array_equal(y, load_and_normalize_sequence([second]))
    assert x[0, 0, 0, 0].tolist() == pytest.approx([10.0 / 80.0, 0.25])
    assert np.all(weights == 0.75)


def test_unobserved_cells_keep_zero_coverage(make_model_frame):
    frame = make_model_frame(dbz=0.0, coverage=0.0)
    reflectivity, coverage = load_frame(frame)
    assert np.all(reflectivity == 0.0)
    assert np.all(coverage == 0.0)


def test_missing_coverage_fails_instead_of_becoming_fully_observed(make_model_frame):
    frame = make_model_frame()
    coverage_path(frame).unlink()
    with pytest.raises(ProvenanceMismatchError, match="missing"):
        load_frame(frame)


def test_missing_manifest_fails_instead_of_accepting_legacy_data(make_model_frame):
    frame = make_model_frame()
    manifest_path(frame).unlink()
    with pytest.raises(ProvenanceMismatchError, match="no provenance manifest"):
        load_frame(frame)


@pytest.mark.parametrize("shape", [(1, 1, 1), (17, 120, 120), (1, 16, 120, 120)])
def test_wrong_tensor_shapes_are_never_padded_cropped_or_squeezed(make_model_frame, shape):
    frame = make_model_frame(shape=shape)
    with pytest.raises(ValueError, match="Expected tensor shape"):
        load_frame(frame)


@pytest.mark.parametrize("dbz", [np.nan, np.inf, -np.inf, -1.0, 81.0])
def test_nonfinite_and_out_of_range_reflectivity_is_rejected(make_model_frame, dbz):
    frame = make_model_frame(dbz=dbz)
    with pytest.raises(ValueError, match="finite|within"):
        load_frame(frame)


@pytest.mark.parametrize("coverage", [np.nan, np.inf, -0.1, 1.1])
def test_invalid_coverage_is_rejected(make_model_frame, coverage):
    frame = make_model_frame(coverage=coverage)
    with pytest.raises(ValueError, match="finite|within"):
        load_frame(frame)


@pytest.mark.parametrize("companion", [False, True])
def test_altered_tensor_or_mask_is_rejected_by_hash(make_model_frame, companion):
    frame = make_model_frame()
    path = coverage_path(frame) if companion else frame
    with path.open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ProvenanceMismatchError, match="content no longer matches"):
        load_frame(frame)


def test_manifest_must_record_the_coverage_companion(make_model_frame):
    frame = make_model_frame()
    change_manifest(frame, lambda payload: payload.update(related_artifacts=[]))
    with pytest.raises(ProvenanceMismatchError, match="Required coverage companion"):
        load_frame(frame)


def test_incompatible_processing_settings_are_rejected(make_model_frame):
    frame = make_model_frame()
    change_manifest(frame, lambda payload: payload["provenance"]["settings"].update(pooling="dbz_mean"))
    with pytest.raises(ProvenanceMismatchError, match="processing settings"):
        load_frame(frame)


def test_source_timestamp_must_match_tensor_name(make_model_frame):
    frame = make_model_frame()
    change_manifest(frame, lambda payload: payload["provenance"]["source"].update(name="another_scan.nc"))
    with pytest.raises(ProvenanceMismatchError, match="source identity"):
        load_frame(frame)


def test_corrupt_files_stop_discovery_instead_of_being_silently_filtered(make_model_frame, tmp_path):
    frame = make_model_frame()
    coverage_path(frame).unlink()
    with pytest.raises(ProvenanceMismatchError):
        get_sorted_files(tmp_path)


def test_coverage_shape_must_match_even_with_a_valid_manifest(make_model_frame):
    frame = make_model_frame()
    companion = coverage_path(frame)
    np.save(companion, np.zeros((1, 1, 1), dtype=np.float32))
    provenance = json.loads(manifest_path(frame).read_text(encoding="utf-8"))["provenance"]
    write_artifact_manifest(frame, provenance, [companion])
    with pytest.raises(ValueError, match="Expected tensor shape"):
        load_frame(frame)


def test_numpy_archive_disguised_as_frame_is_rejected(make_model_frame):
    frame = make_model_frame()
    with frame.open("wb") as handle:
        np.savez(handle, data=np.zeros((1, 1, 1)))
    provenance = json.loads(manifest_path(frame).read_text(encoding="utf-8"))["provenance"]
    write_artifact_manifest(frame, provenance, [coverage_path(frame)])
    with pytest.raises(ValueError, match="single NumPy array"):
        load_frame(frame)


def test_model_sequences_reject_wrong_timestamps_before_loading(make_model_frame):
    first = make_model_frame(0)
    second = make_model_frame(2)
    with pytest.raises(ValueError, match="cadence"):
        next(sequence_generator([[first, second]], 1, 1))
