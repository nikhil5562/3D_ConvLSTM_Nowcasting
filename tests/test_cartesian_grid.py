import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyart
import xarray as xr

from nowcasting.data.cartesian_grid import write_cartesian_grid
from nowcasting.data.cartesian_grid import _validate_written_grid
from nowcasting.data.l2b_reader import NETCDF_FILL_VALUE


class CartesianGridTests(unittest.TestCase):
    def test_written_grid_validation_rejects_wrong_fill_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.nc"
            dataset = xr.Dataset(
                {"DBZ": (("time", "z", "y", "x"), np.zeros((1, 2, 3, 3)))},
            )
            dataset["DBZ"].encoding["_FillValue"] = 0.0
            dataset.to_netcdf(path)
            with patch(
                "nowcasting.data.cartesian_grid.FULL_CUBE_SHAPE", (2, 3, 3)
            ), self.assertRaisesRegex(ValueError, "fill value"):
                _validate_written_grid(path)

    def test_atomic_grid_write_serializes_nonphysical_fill_value(self):
        radar = pyart.testing.make_empty_ppi_radar(10, 10, 1)
        values = np.ma.masked_array(
            np.full((10, 10), 15.0, dtype=np.float32),
            mask=False,
        )
        values.mask[0, 0] = True
        radar.fields["DBZ"] = {
            "data": values,
            "units": "dBZ",
            "standard_name": "equivalent_reflectivity_factor",
            "_FillValue": 0.0,
        }

        with tempfile.TemporaryDirectory() as tmp, patch(
            "nowcasting.data.cartesian_grid.FULL_CUBE_SHAPE",
            (2, 3, 3),
        ), patch("nowcasting.data.cartesian_grid.ALT_MAX_M", 1000.0), patch(
            "nowcasting.data.cartesian_grid.HALF_HORIZ_M",
            1000.0,
        ):
            output = Path(tmp) / "cube.nc"
            write_cartesian_grid(radar, output, roi_func="constant", constant_roi=1000.0)
            with xr.open_dataset(output, decode_times=False) as dataset:
                fill_value = dataset["DBZ"].encoding.get("_FillValue")
            temporary_files = list(Path(tmp).glob("*.tmp.nc"))

        self.assertEqual(fill_value, NETCDF_FILL_VALUE)
        self.assertEqual(temporary_files, [])


if __name__ == "__main__":
    unittest.main()
