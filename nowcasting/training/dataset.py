"""Finite TensorFlow datasets with explicit cardinality and repeatable epochs."""

import random

import tensorflow as tf

from nowcasting.config import (
    INPUT_CHANNELS, INPUT_LENGTH, OUTPUT_LENGTH, RANDOM_SEED, TARGET_CHANNELS,
    TARGET_SHAPE,
)
from nowcasting.data.sequences import sequence_generator
from nowcasting.training.settings import BATCH_SIZE


def create_dataset(sequences, training=False, batch_size=BATCH_SIZE, max_batches=None):
    """Build a finite epoch; optional limits apply anew to both fit and validation."""
    if batch_size < 1 or (max_batches is not None and max_batches < 1):
        raise ValueError("Batch size and optional batch limits must be positive.")
    sequences = tuple(tuple(sequence) for sequence in sequences)
    epoch_rng = random.Random(RANDOM_SEED)

    def generate():
        yield from sequence_generator(
            sequences, INPUT_LENGTH, OUTPUT_LENGTH,
            shuffle=training, seed=epoch_rng.randrange(2**31),
        )

    output_signature = (
        tf.TensorSpec((INPUT_LENGTH, *TARGET_SHAPE, INPUT_CHANNELS), tf.float32),
        tf.TensorSpec((OUTPUT_LENGTH, *TARGET_SHAPE, TARGET_CHANNELS), tf.float32),
        tf.TensorSpec((OUTPUT_LENGTH, *TARGET_SHAPE, TARGET_CHANNELS), tf.float32),
    )
    dataset = tf.data.Dataset.from_generator(generate, output_signature=output_signature)
    dataset = dataset.apply(tf.data.experimental.assert_cardinality(len(sequences)))
    dataset = dataset.batch(batch_size, drop_remainder=False)
    if max_batches is not None:
        dataset = dataset.take(max_batches)
    options = tf.data.Options()
    options.experimental_deterministic = True
    return dataset.with_options(options).prefetch(tf.data.AUTOTUNE)
