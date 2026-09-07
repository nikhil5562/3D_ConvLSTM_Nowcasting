from datetime import datetime, timedelta
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

from nowcasting.data import model_data
from nowcasting.data.provenance import write_artifact_manifest
from nowcasting.data.tensorize import _tensor_provenance


@pytest.fixture
def make_model_frame(tmp_path):
    """Write a complete synthetic tensor product through the real manifest writer."""
    def make(index=0, dbz=20.0, coverage=0.75, shape=None):
        timestamp = datetime(2025, 1, 1) + timedelta(minutes=15 * index)
        stem = f"RCTLS_{timestamp:%d%b%Y_%H%M%S}_L2B_STD"
        frame = tmp_path / f"{stem}.npy"
        cube = tmp_path / f"{stem}_gridded.nc"
        cube.write_bytes(b"synthetic source cube")
        companion = model_data.coverage_path(frame)
        companion.parent.mkdir(exist_ok=True)
        frame_shape = model_data.TARGET_SHAPE if shape is None else shape
        np.save(frame, np.full(frame_shape, dbz, dtype=np.float32))
        np.save(companion, np.full(frame_shape, coverage, dtype=np.float32))
        write_artifact_manifest(frame, _tensor_provenance(cube), [companion])
        return frame
    return make
