import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import xarray as xr

from nowcasting.data.tensorize import (
    MODEL_TENSOR_SHAPE,
    _as_3d_cube,
    coverage_output_path,
    downsample_dbz_to_model_tensor,
    load_dbz_cube,
    tensor_output_path,
    tensorize_cube,
)
from nowcasting.data.provenance import ProvenanceMismatchError, manifest_path


class TensorizeTests(unittest.TestCase):
    def test_downsamples_full_cube_to_current_model_shape(self):
        cube = np.zeros((81, 481, 481), dtype=np.float32)

        tensor = downsample_dbz_to_model_tensor(cube)

        self.assertEqual(tensor.shape, MODEL_TENSOR_SHAPE)
        self.assertEqual(tensor.dtype, np.float32)

    def test_downsamples_by_averaging_in_linear_reflectivity_space(self):
        cube = np.zeros((81, 481, 481), dtype=np.float32)
        cube[0:4, 0:4, 0:4] = 10.0
        cube[0:4, 0:4, 0:2] = 20.0

        tensor = downsample_dbz_to_model_tensor(cube)

        expected_linear = (32 * (10 ** (20.0 / 10.0)) + 32 * (10 ** (10.0 / 10.0))) / 64
        expected_dbz = 10.0 * np.log10(expected_linear)
        self.assertAlmostEqual(float(tensor[0, 0, 0]), float(expected_dbz), places=5)

    def test_trims_outer_481st_row_and_column_before_pooling(self):
        cube = np.zeros((81, 481, 481), dtype=np.float32)
        cube[:, 480, :] = 80.0
        cube[:, :, 480] = 80.0

        tensor = downsample_dbz_to_model_tensor(cube)

        self.assertTrue(np.allclose(tensor, 0.0))

    def test_excludes_missing_voxels_and_returns_coverage_fraction(self):
        cube = np.zeros((81, 481, 481), dtype=np.float32)
        cube[0:4, 0:4, 0:4] = np.nan
        cube[0, 0, 0] = 20.0

        tensor, coverage = downsample_dbz_to_model_tensor(cube, return_coverage=True)

        self.assertAlmostEqual(float(tensor[0, 0, 0]), 20.0, places=5)
        self.assertAlmostEqual(float(coverage[0, 0, 0]), 1.0 / 64.0, places=7)
        self.assertEqual(coverage.dtype, np.float32)

    def test_preserves_mask_before_array_conversion(self):
        masked = np.ma.masked_array(
            np.array([[[42.0]]], dtype=np.float32),
            mask=np.array([[[True]]]),
        )

        cube = _as_3d_cube(masked)

        self.assertTrue(np.isnan(cube[0, 0, 0]))

    def test_loads_dbz_from_pyart_grid_netcdf_schema(self):
        cube = np.zeros((1, 81, 481, 481), dtype=np.float32)
        cube[0, 0, 0, 0] = 12.5

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "RCTLS_09JUN2026_050131_L2B_STD_gridded.nc"
            ds = xr.Dataset(
                {"DBZ": (("time", "z", "y", "x"), cube)},
                coords={
                    "time": np.array([0.0]),
                    "z": np.arange(81, dtype=np.float32) * 250.0,
                    "y": np.arange(481, dtype=np.float32),
                    "x": np.arange(481, dtype=np.float32),
                },
            )
            ds.to_netcdf(path)
            ds.close()

            loaded = load_dbz_cube(path)

        self.assertEqual(loaded.shape, (81, 481, 481))
        self.assertEqual(float(loaded[0, 0, 0]), 12.5)

    def test_tensor_output_path_removes_gridded_suffix(self):
        output = tensor_output_path(
            Path("RCTLS_09JUN2026_050131_L2B_STD_gridded.nc"),
            Path("processed_data"),
        )

        self.assertEqual(output, Path("processed_data/RCTLS_09JUN2026_050131_L2B_STD.npy"))

    def test_tensorize_writes_atomic_coverage_and_provenance_sidecars(self):
        tensor = np.full(MODEL_TENSOR_SHAPE, 12.0, dtype=np.float32)
        coverage = np.full(MODEL_TENSOR_SHAPE, 0.75, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "RCTLS_09JUN2026_050131_L2B_STD_gridded.nc"
            source.write_bytes(b"source cube")
            output_dir = root / "processed"

            with patch("nowcasting.data.tensorize.load_dbz_cube", return_value=np.zeros((1, 1, 1))), patch(
                "nowcasting.data.tensorize.downsample_dbz_to_model_tensor",
                return_value=(tensor, coverage),
            ):
                output = tensorize_cube(source, output_dir)

            coverage_path = coverage_output_path(source, output_dir)
            self.assertTrue(output.is_file())
            self.assertTrue(coverage_path.is_file())
            self.assertTrue(manifest_path(output).is_file())
            np.testing.assert_array_equal(np.load(output), tensor)
            np.testing.assert_array_equal(np.load(coverage_path), coverage)

            # A complete, unchanged artifact is safely reusable without recomputation.
            self.assertEqual(tensorize_cube(source, output_dir), output)

            source.write_bytes(b"changed source cube")
            with self.assertRaisesRegex(ProvenanceMismatchError, "different source data or settings"):
                tensorize_cube(source, output_dir)

    def test_refuses_existing_tensor_without_matching_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "RCTLS_09JUN2026_050131_L2B_STD_gridded.nc"
            source.write_bytes(b"source cube")
            output_dir = root / "processed"
            output = tensor_output_path(source, output_dir)
            output.parent.mkdir(parents=True)
            np.save(output, np.zeros(MODEL_TENSOR_SHAPE, dtype=np.float32))

            with self.assertRaisesRegex(ProvenanceMismatchError, "no provenance manifest"):
                tensorize_cube(source, output_dir)


if __name__ == "__main__":
    unittest.main()
