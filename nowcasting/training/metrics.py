"""Stateful forecast metrics accumulated over observed radar coverage."""

import tensorflow as tf

from nowcasting.config import DBZ_MAX, OUTPUT_LENGTH


@tf.keras.utils.register_keras_serializable(package="nowcasting")
class ContingencyMetric(tf.keras.metrics.Metric):
    """Globally accumulated CSI, POD, or FAR at one threshold and lead."""

    def __init__(self, score, threshold_dbz, lead_index=None, name=None, **kwargs):
        if score not in {"csi", "pod", "far"}:
            raise ValueError(f"Unsupported contingency score: {score}")
        self.score = score
        self.threshold_dbz = float(threshold_dbz)
        self.lead_index = lead_index
        if name is None:
            suffix = "" if lead_index is None else f"_t{lead_index + 1:02d}"
            name = f"{score}{int(threshold_dbz)}{suffix}"
        super().__init__(name=name, **kwargs)
        self.hits = self.add_weight(name="hits", initializer="zeros", dtype=tf.float64)
        self.misses = self.add_weight(name="misses", initializer="zeros", dtype=tf.float64)
        self.false_alarms = self.add_weight(
            name="false_alarms", initializer="zeros", dtype=tf.float64
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.lead_index is not None:
            y_true = tf.gather(y_true, self.lead_index, axis=1)
            y_pred = tf.gather(y_pred, self.lead_index, axis=1)
            if sample_weight is not None and sample_weight.shape.rank != 0:
                sample_weight = tf.gather(sample_weight, self.lead_index, axis=1)

        threshold = tf.cast(self.threshold_dbz / DBZ_MAX, y_true.dtype)
        observed = tf.cast(y_true >= threshold, tf.float64)
        forecast = tf.cast(y_pred >= threshold, tf.float64)
        if sample_weight is None:
            weight = tf.ones_like(observed, dtype=tf.float64)
        else:
            weight = tf.cast(sample_weight, tf.float64)
            weight = tf.broadcast_to(weight, tf.shape(observed))

        self.hits.assign_add(tf.reduce_sum(weight * observed * forecast))
        self.misses.assign_add(tf.reduce_sum(weight * observed * (1.0 - forecast)))
        self.false_alarms.assign_add(
            tf.reduce_sum(weight * (1.0 - observed) * forecast)
        )

    def result(self):
        if self.score == "csi":
            denominator = self.hits + self.misses + self.false_alarms
            numerator = self.hits
        elif self.score == "pod":
            denominator = self.hits + self.misses
            numerator = self.hits
        else:
            denominator = self.hits + self.false_alarms
            numerator = self.false_alarms
        return tf.math.divide_no_nan(numerator, denominator)

    def reset_state(self):
        self.hits.assign(0.0)
        self.misses.assign(0.0)
        self.false_alarms.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "score": self.score,
                "threshold_dbz": self.threshold_dbz,
                "lead_index": self.lead_index,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="nowcasting")
class PredictionStd(tf.keras.metrics.Metric):
    """Coverage-weighted global prediction standard deviation."""

    def __init__(self, name="pred_std", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros", dtype=tf.float64)
        self.total_sq = self.add_weight(
            name="total_sq", initializer="zeros", dtype=tf.float64
        )
        self.count = self.add_weight(name="count", initializer="zeros", dtype=tf.float64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = tf.cast(y_pred, tf.float64)
        if sample_weight is None:
            weight = tf.ones_like(values, dtype=tf.float64)
        else:
            weight = tf.broadcast_to(tf.cast(sample_weight, tf.float64), tf.shape(values))
        self.total.assign_add(tf.reduce_sum(weight * values))
        self.total_sq.assign_add(tf.reduce_sum(weight * tf.square(values)))
        self.count.assign_add(tf.reduce_sum(weight))

    def result(self):
        mean = tf.math.divide_no_nan(self.total, self.count)
        variance = tf.math.divide_no_nan(self.total_sq, self.count) - tf.square(mean)
        return tf.sqrt(tf.maximum(variance, 0.0))

    def reset_state(self):
        self.total.assign(0.0)
        self.total_sq.assign(0.0)
        self.count.assign(0.0)


def build_metrics(include_per_lead=False):
    metrics = [PredictionStd()]
    for threshold_dbz in (20.0, 35.0):
        for score in ("csi", "pod", "far"):
            metrics.append(ContingencyMetric(score, threshold_dbz))
        if include_per_lead:
            for lead_index in range(OUTPUT_LENGTH):
                for score in ("csi", "pod", "far"):
                    metrics.append(
                        ContingencyMetric(score, threshold_dbz, lead_index=lead_index)
                    )
    return metrics
