"""Shared experiment configuration for training, evaluation, and inference."""

import os


INPUT_LENGTH = int(os.environ.get("NOWCAST_INPUT_LENGTH", "10"))
OUTPUT_LENGTH = int(os.environ.get("NOWCAST_OUTPUT_LENGTH", "8"))
NOMINAL_CADENCE_MINUTES = int(os.environ.get("NOWCAST_CADENCE_MINUTES", "15"))

TARGET_SHAPE = (16, 120, 120)
INPUT_CHANNELS = 2  # normalized reflectivity plus observed-coverage fraction
TARGET_CHANNELS = 1
DBZ_MIN = 0.0
DBZ_MAX = 80.0

FORECAST_LEAD_MINUTES = tuple(
    NOMINAL_CADENCE_MINUTES * step for step in range(1, OUTPUT_LENGTH + 1)
)
FORECAST_HORIZON_MINUTES = FORECAST_LEAD_MINUTES[-1]

if INPUT_LENGTH < 1 or OUTPUT_LENGTH < 1:
    raise ValueError("NOWCAST_INPUT_LENGTH and NOWCAST_OUTPUT_LENGTH must be positive.")
if NOMINAL_CADENCE_MINUTES < 1:
    raise ValueError("NOWCAST_CADENCE_MINUTES must be positive.")
