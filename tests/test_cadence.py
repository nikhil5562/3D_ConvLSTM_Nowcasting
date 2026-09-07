import json
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("cadence", [5, 15])
def test_configured_cadence_agrees_across_preflight_windows_and_forecast_labels(cadence):
    script = """
import json
from datetime import datetime, timedelta
from nowcasting.config import NOMINAL_CADENCE_MINUTES, FORECAST_LEAD_MINUTES
from nowcasting.data.l2b_reader import diagnose_sequence_capacity
from nowcasting.data.sequences import get_sequences
from nowcasting.data.timestamps import find_contiguous_window, validate_sequence_paths

start = datetime(2025, 1, 1)
records = []
for index in range(4):
    timestamp = start + timedelta(minutes=NOMINAL_CADENCE_MINUTES * index)
    records.append({"time": timestamp, "echo": 1.0,
        "path": f"RCTLS_{timestamp:%d%b%Y_%H%M%S}_L2B_STD.npy"})
paths = [record["path"] for record in records]
validate_sequence_paths(paths, 4)
capacity = diagnose_sequence_capacity([path.replace('.npy', '.nc') for path in paths], 4)
print(json.dumps({"preflight": capacity["sequence_count"],
    "training": len(get_sequences(records, 2, 2)),
    "diagnostic": len(find_contiguous_window(records, 4)),
    "leads": FORECAST_LEAD_MINUTES}))
"""
    environment = dict(os.environ, NOWCAST_CADENCE_MINUTES=str(cadence),
                       NOWCAST_INPUT_LENGTH="2", NOWCAST_OUTPUT_LENGTH="2", PYART_QUIET="1")
    result = subprocess.run([sys.executable, "-c", script], env=environment,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == {
        "preflight": 1, "training": 1, "diagnostic": 4, "leads": [cadence, cadence * 2],
    }


@pytest.mark.parametrize("setting", ["NOWCAST_OUTPUT_LENGTH", "NOWCAST_INPUT_LENGTH", "NOWCAST_CADENCE_MINUTES"])
def test_invalid_configuration_reports_value_error_before_derived_values(setting):
    environment = dict(os.environ, **{setting: "0"})
    result = subprocess.run([sys.executable, "-c", "import nowcasting.config"],
                            env=environment, capture_output=True, text=True)
    assert result.returncode != 0
    assert "ValueError" in result.stderr
    assert "must be positive" in result.stderr
    assert "IndexError" not in result.stderr
