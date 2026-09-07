"""Coverage-normalized reflectivity loss used in training and evaluation."""

import tensorflow as tf

from nowcasting.config import DBZ_MAX


def weighted_loss(y_true, y_pred):
    """Reflectivity-weighted MAE+MSE.

    FIX 3: with the old weights (1, 3, 8, 15) about 75% of the loss mass came
    from near-empty cells, so predicting all-zero was near-optimal and the
    model collapsed. Measured class frequencies are ~90.1% below 15 dBZ, 9.8%
    in 15-35, 0.08% in 35-45 and 0.03% above 45 dBZ, so the heavy classes need
    far larger weights to contribute a meaningful share of the gradient.
    """
    t1 = 15.0 / DBZ_MAX
    t2 = 35.0 / DBZ_MAX
    t3 = 45.0 / DBZ_MAX

    w1, w2, w3, w4 = 1.0, 10.0, 50.0, 100.0

    mask = tf.ones_like(y_true) * w1
    mask = tf.where((y_true >= t1) & (y_true < t2), tf.ones_like(y_true) * w2, mask)
    mask = tf.where((y_true >= t2) & (y_true < t3), tf.ones_like(y_true) * w3, mask)
    mask = tf.where(y_true >= t3, tf.ones_like(y_true) * w4, mask)

    # Return an unreduced field. Keras applies the target-coverage sample
    # weights before reduction, so unobserved radar cells contribute nothing.
    return (tf.abs(y_pred - y_true) + tf.square(y_pred - y_true)) * mask


@tf.keras.utils.register_keras_serializable(package="nowcasting")
class CoverageWeightedLoss(tf.keras.losses.Loss):
    """Reflectivity loss normalized by the amount of observed radar coverage."""

    def __init__(
        self,
        name="coverage_weighted_loss",
        reduction="mean_with_sample_weight",
    ):
        super().__init__(name=name, reduction=reduction)

    def call(self, y_true, y_pred):
        return weighted_loss(y_true, y_pred)
