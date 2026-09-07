"""Held-out model and persistence evaluation at each forecast lead."""

import math

import numpy as np
import tensorflow as tf

from nowcasting.config import DBZ_MAX, FORECAST_LEAD_MINUTES, OUTPUT_LENGTH
from nowcasting.training.losses import weighted_loss


def _new_evaluation_accumulator():
    return {
        "loss_numerator": 0.0,
        "weight_sum": 0.0,
        "pred_sum": 0.0,
        "pred_sq_sum": 0.0,
        "contingency": {
            threshold: np.zeros((OUTPUT_LENGTH, 3), dtype=np.float64)
            for threshold in (20.0, 35.0)
        },
    }


def _accumulate_evaluation(accumulator, y_true, y_pred, coverage):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    coverage = np.asarray(coverage, dtype=np.float64)
    if y_true.shape != y_pred.shape or y_true.shape != coverage.shape:
        raise ValueError(
            f"Evaluation shapes differ: truth={y_true.shape}, prediction={y_pred.shape}, "
            f"coverage={coverage.shape}"
        )

    per_cell_loss = np.asarray(weighted_loss(y_true, y_pred), dtype=np.float64)
    accumulator["loss_numerator"] += float(np.sum(per_cell_loss * coverage))
    accumulator["weight_sum"] += float(np.sum(coverage))
    accumulator["pred_sum"] += float(np.sum(y_pred * coverage))
    accumulator["pred_sq_sum"] += float(np.sum(np.square(y_pred) * coverage))

    reduction_axes = tuple(axis for axis in range(y_true.ndim) if axis != 1)
    for threshold_dbz, counts in accumulator["contingency"].items():
        threshold = threshold_dbz / DBZ_MAX
        observed = y_true >= threshold
        forecast = y_pred >= threshold
        counts[:, 0] += np.sum(coverage * observed * forecast, axis=reduction_axes)
        counts[:, 1] += np.sum(coverage * observed * ~forecast, axis=reduction_axes)
        counts[:, 2] += np.sum(coverage * ~observed * forecast, axis=reduction_axes)


def _scores_from_counts(hits, misses, false_alarms):
    def divide(numerator, denominator):
        return float(numerator / denominator) if denominator else 0.0

    return {
        "csi": divide(hits, hits + misses + false_alarms),
        "pod": divide(hits, hits + misses),
        "far": divide(false_alarms, hits + false_alarms),
        "hits": float(hits),
        "misses": float(misses),
        "false_alarms": float(false_alarms),
    }


def _finalize_evaluation(accumulator):
    weight_sum = accumulator["weight_sum"]
    mean = accumulator["pred_sum"] / weight_sum if weight_sum else 0.0
    variance = accumulator["pred_sq_sum"] / weight_sum - mean * mean if weight_sum else 0.0
    result = {
        "coverage_weighted_loss": (
            accumulator["loss_numerator"] / weight_sum if weight_sum else 0.0
        ),
        "prediction_std": math.sqrt(max(variance, 0.0)),
        "evaluated_coverage_weight": weight_sum,
        "overall": {},
        "per_lead": [],
    }
    for lead_index, lead_minutes in enumerate(FORECAST_LEAD_MINUTES):
        lead_result = {"lead_index": lead_index + 1, "lead_minutes": lead_minutes}
        for threshold_dbz, counts in accumulator["contingency"].items():
            lead_result[f"{int(threshold_dbz)}_dbz"] = _scores_from_counts(*counts[lead_index])
        result["per_lead"].append(lead_result)

    for threshold_dbz, counts in accumulator["contingency"].items():
        result["overall"][f"{int(threshold_dbz)}_dbz"] = _scores_from_counts(
            *np.sum(counts, axis=0)
        )
    return result


def evaluate_test_set(model, test_dataset):
    """Evaluate the trained model and persistence on the same held-out data."""
    model_accumulator = _new_evaluation_accumulator()
    persistence_accumulator = _new_evaluation_accumulator()
    batches = 0
    for x_batch, y_batch, coverage_batch in test_dataset:
        y_prediction = model(x_batch, training=False)
        persistence = tf.repeat(x_batch[:, -1:, ..., :1], OUTPUT_LENGTH, axis=1)
        _accumulate_evaluation(model_accumulator, y_batch, y_prediction, coverage_batch)
        _accumulate_evaluation(
            persistence_accumulator, y_batch, persistence, coverage_batch
        )
        batches += 1
    if batches == 0:
        raise RuntimeError("Held-out test dataset produced no batches.")
    model_results = _finalize_evaluation(model_accumulator)
    persistence_results = _finalize_evaluation(persistence_accumulator)
    persistence_loss = persistence_results["coverage_weighted_loss"]
    return {
        "model": model_results,
        "persistence": persistence_results,
        "loss_skill_vs_persistence": (
            1.0 - model_results["coverage_weighted_loss"] / persistence_loss
            if persistence_loss > 0.0
            else None
        ),
        "batches": batches,
    }
