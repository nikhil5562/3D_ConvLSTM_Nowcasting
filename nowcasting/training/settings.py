"""Training controls, independent of TensorFlow imports."""

import os


BATCH_SIZE = int(os.environ.get("NOWCAST_BATCH_SIZE", "4"))
EPOCHS = int(os.environ.get("NOWCAST_EPOCHS", "20"))
LEARNING_RATE = 0.0001
GRAD_CLIP_NORM = 1.0

if BATCH_SIZE < 1 or EPOCHS < 1:
    raise ValueError("NOWCAST_BATCH_SIZE and NOWCAST_EPOCHS must be positive.")
