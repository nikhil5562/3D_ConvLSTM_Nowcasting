"""Cadence-aware windows and independent calendar-day splits."""

import random
from collections import defaultdict

import numpy as np

from nowcasting.config import RANDOM_SEED
from nowcasting.data.model_data import load_frame
from nowcasting.data.timestamps import is_contiguous, parse_timestamp, validate_sequence_paths

MIN_ECHO_FRACTION = 0.02


def sequence_generator(sequences, input_len, output_len, shuffle=False, seed=RANDOM_SEED):
    """Yield normalized inputs, reflectivity targets, and target coverage weights."""
    ordered_sequences = list(sequences)
    if shuffle:
        random.Random(seed).shuffle(ordered_sequences)
    for paths in ordered_sequences:
        validate_sequence_paths(paths, input_len + output_len)
        pairs = [load_frame(path) for path in paths]
        reflectivity = np.stack([pair[0] for pair in pairs])
        coverage = np.stack([pair[1] for pair in pairs])
        x = np.concatenate([reflectivity[:input_len], coverage[:input_len]], axis=-1)
        yield x, reflectivity[input_len:], coverage[input_len:]


def get_sequences(file_list, input_len, output_len):
    sequences = []
    seq_len = input_len + output_len

    if len(file_list) < seq_len:
        return sequences

    for i in range(len(file_list) - seq_len + 1):
        window = file_list[i : i + seq_len]
        if not is_contiguous(record["time"] for record in window):
            continue

        # Select on observations only. Looking at future target echo coverage
        # would leak ground truth and systematically remove initiation/decay.
        if min(f.get("echo", 0.0) for f in window[:input_len]) < MIN_ECHO_FRACTION:
            continue

        sequences.append(
            {
                "start": i,
                "paths": [f["path"] for f in window],
            }
        )

    return sequences


def _allocate_split_units(units, train_ratio=0.8, val_ratio=0.1):
    """Chronologically allocate whole days, keeping at least one per split."""
    units = list(units)
    if len(units) < 3:
        raise ValueError(
            "At least three independent radar days are required for "
            "train/validation/test splitting."
        )
    n_train = min(max(1, int(len(units) * train_ratio)), len(units) - 2)
    n_val = min(max(1, int(len(units) * val_ratio)), len(units) - n_train - 1)
    return (
        units[:n_train],
        units[n_train : n_train + n_val],
        units[n_train + n_val :],
    )


def split_files_by_day(file_list, train_ratio=0.8, val_ratio=0.1):
    """Split complete calendar days before constructing temporal windows.

    This is intentionally chronological: validation and test data occur after
    training data. A window is built only from files belonging to one split,
    so no frame or day can leak across the boundaries.
    """
    by_day = defaultdict(list)
    for item in file_list:
        by_day[item["time"].date().isoformat()].append(item)

    train_days, val_days, test_days = _allocate_split_units(
        sorted(by_day), train_ratio=train_ratio, val_ratio=val_ratio
    )

    def flatten(days):
        return [item for day in days for item in by_day[day]]

    return (
        flatten(train_days),
        flatten(val_days),
        flatten(test_days),
        {"train": train_days, "validation": val_days, "test": test_days},
    )


def create_day_independent_splits(
    file_list, input_len, output_len, train_ratio=0.8, val_ratio=0.1
):
    """Split days first, then construct sequences independently per day."""
    by_day = defaultdict(list)
    for item in file_list:
        by_day[item["time"].date().isoformat()].append(item)

    sequences_by_day = {}
    excluded_days = []
    for day in sorted(by_day):
        records = get_sequences(by_day[day], input_len, output_len)
        if records:
            sequences_by_day[day] = [record["paths"] for record in records]
        else:
            excluded_days.append(day)

    train_days, val_days, test_days = _allocate_split_units(
        sorted(sequences_by_day), train_ratio=train_ratio, val_ratio=val_ratio
    )
    split_days = {
        "train": train_days,
        "validation": val_days,
        "test": test_days,
        "excluded_no_sequences": excluded_days,
    }
    flatten = lambda days: [seq for day in days for seq in sequences_by_day[day]]
    return flatten(train_days), flatten(val_days), flatten(test_days), split_days


def split_sequences_no_leakage(
    all_sequences, input_len, output_len, train_ratio=0.8, val_ratio=0.1
):
    """Compatibility wrapper that allocates already-built windows by day.

    New training code uses :func:`create_day_independent_splits`, which splits
    before windows are built. This wrapper remains for older callers and
    rejects cross-midnight windows rather than assigning them ambiguously.
    """
    by_day = defaultdict(list)
    for sequence in all_sequences:
        paths = sequence["paths"] if isinstance(sequence, dict) else sequence
        days = {parse_timestamp(path).date().isoformat() for path in paths}
        if len(days) == 1:
            by_day[next(iter(days))].append(list(paths))
    if len(by_day) < 3:
        return [], [], []
    train_days, val_days, test_days = _allocate_split_units(
        sorted(by_day), train_ratio=train_ratio, val_ratio=val_ratio
    )
    flatten = lambda days: [seq for day in days for seq in by_day[day]]
    return flatten(train_days), flatten(val_days), flatten(test_days)


def _report_split_overlap(train_sequences, val_sequences, test_sequences):
    train_files = {path for seq in train_sequences for path in seq}
    val_files = {path for seq in val_sequences for path in seq}
    test_files = {path for seq in test_sequences for path in seq}

    overlap_train_val = len(train_files & val_files)
    overlap_train_test = len(train_files & test_files)
    overlap_val_test = len(val_files & test_files)

    print(
        "File overlap counts (must be zero): "
        f"train-val={overlap_train_val}, "
        f"train-test={overlap_train_test}, "
        f"val-test={overlap_val_test}"
    )

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise RuntimeError("Data leakage detected across train/val/test splits.")

    def sequence_days(sequences):
        return {parse_timestamp(path).date().isoformat() for seq in sequences for path in seq}

    train_days = sequence_days(train_sequences)
    val_days = sequence_days(val_sequences)
    test_days = sequence_days(test_sequences)
    if train_days & val_days or train_days & test_days or val_days & test_days:
        raise RuntimeError("Radar-day leakage detected across train/val/test splits.")
