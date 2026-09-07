import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

from nowcasting.data.l2b_reader import (
    NETCDF_FILL_VALUE,
    _fix_dbz_fill_value,
    _fix_sweep_ray_indices,
    diagnose_sequence_capacity,
    discover_l2b_candidates,
    is_l2b_standard_file,
    is_probable_full_volume_file,
    is_full_volume_radar,
    parse_timestamp_from_filename,
)


class L2BReaderTests(unittest.TestCase):
    def test_parses_rctls_timestamp_from_filename(self):
        timestamp = parse_timestamp_from_filename("RCTLS_09JUN2026_050131_L2B_STD.nc")

        self.assertEqual(timestamp, datetime(2026, 6, 9, 5, 1, 31))

    def test_l2b_standard_match_is_strict(self):
        self.assertTrue(is_l2b_standard_file(Path("RCTLS_09JUN2026_050131_L2B_STD.nc")))
        self.assertFalse(is_l2b_standard_file(Path("RCTLS_09JUN2026_050131_L2C_STD.nc")))
        self.assertFalse(is_l2b_standard_file(Path("clean_grid_RCTLS_09JUN2026_050131_L2B_STD.nc")))

    def test_discovers_only_probable_full_volume_l2b_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full_scan = root / "RCTLS_09JUN2026_050131_L2B_STD.nc"
            short_scan = root / "RCTLS_09JUN2026_051411_L2B_STD.nc"
            l2c_file = root / "RCTLS_09JUN2026_050131_L2C_STD.nc"

            full_scan.write_bytes(b"0" * 128)
            short_scan.write_bytes(b"0" * 16)
            l2c_file.write_bytes(b"0" * 256)

            candidates = discover_l2b_candidates(root, min_size_mb=0.0001)

        self.assertEqual(candidates, [full_scan])

    def test_full_volume_radar_requires_minimum_sweeps(self):
        class RadarStub:
            def __init__(self, nsweeps):
                self.nsweeps = nsweeps

        self.assertTrue(is_full_volume_radar(RadarStub(11)))
        self.assertTrue(is_full_volume_radar(RadarStub(12)))
        self.assertFalse(is_full_volume_radar(RadarStub(3)))

    def test_size_prefilter_is_configurable(self):
        with tempfile.TemporaryDirectory() as tmp:
            scan = Path(tmp) / "RCTLS_09JUN2026_050131_L2B_STD.nc"
            scan.write_bytes(b"0" * 64)

            self.assertTrue(is_probable_full_volume_file(scan, min_size_mb=0.00001))
            self.assertFalse(is_probable_full_volume_file(scan, min_size_mb=1.0))

    def test_dbz_schema_fix_preserves_source_mask_and_uses_safe_fill(self):
        source = np.ma.masked_array(
            np.array([[0.0, 12.0], [-40.0, np.nan]], dtype=np.float32),
            mask=np.array([[True, False], [False, False]]),
        )

        class RadarStub:
            fields = {"DBZ": {"data": source, "_FillValue": 0.0}}

        radar = RadarStub()
        _fix_dbz_fill_value(radar)

        field = radar.fields["DBZ"]
        self.assertTrue(np.array_equal(np.ma.getmaskarray(field["data"]), [[True, False], [True, True]]))
        self.assertEqual(field["_FillValue"], NETCDF_FILL_VALUE)
        self.assertEqual(field["missing_value"], NETCDF_FILL_VALUE)

    def test_rebuild_sweep_indices_requires_even_complete_partition(self):
        class RadarStub:
            nsweeps = 2
            nrays = 23
            sweep_start_ray_index = {"data": np.array([0, 0])}
            sweep_end_ray_index = {"data": np.array([0, 0])}

        with self.assertRaisesRegex(ValueError, "not divisible"):
            _fix_sweep_ray_indices(RadarStub())

    def test_rebuilds_sweep_indices_and_covers_last_ray(self):
        class RadarStub:
            nsweeps = 2
            nrays = 24
            sweep_start_ray_index = {"data": np.array([0, 0])}
            sweep_end_ray_index = {"data": np.array([0, 0])}

        radar = RadarStub()
        _fix_sweep_ray_indices(radar)

        np.testing.assert_array_equal(radar.sweep_start_ray_index["data"], [0, 12])
        np.testing.assert_array_equal(radar.sweep_end_ray_index["data"], [11, 23])

    def test_sequence_capacity_reports_short_contiguous_runs(self):
        paths = [
            f"RCTLS_09JUN2026_{hour:02d}{minute:02d}00_L2B_STD.nc"
            for hour, minute in [(0, 0), (0, 15), (0, 30), (1, 30), (1, 45)]
        ]

        result = diagnose_sequence_capacity(paths, frames_per_sequence=4)

        self.assertEqual(result["run_lengths"], [3, 2])
        self.assertEqual(result["longest_run"], 3)
        self.assertEqual(result["sequence_count"], 0)


if __name__ == "__main__":
    unittest.main()
