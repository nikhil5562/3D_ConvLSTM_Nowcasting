"""Adapted 3D-ConvLSTM encoder-forecaster for volumetric radar nowcasting.

The local implementation includes unit forget-gate bias, layer-normalised gate
pre-activations, and a sigmoid output initialized near the observed base rate.
These changes address signal attenuation observed in the original local port.

Unlike Sun et al.'s reference implementation, the local forecaster receives a
repeated final encoded frame in addition to its recurrent state. This is an
architectural adaptation, not a correction to the reference's intentional
zero-input design, and requires a controlled ablation before publication.

Optional: `use_skip=True` adds an encoder->decoder skip connection. It is OFF
by default because the encoder frames are PAST and the decoder frames are
FUTURE; wiring them together makes it easy for the model to learn persistence
(copy the last input) and score well on MSE while being meteorologically
useless. Enable only if you evaluate against a persistence baseline.
"""

import os

import tensorflow as tf
from tensorflow.keras import layers, Model

# Legacy-corpus normalized reflectivity mean was ~0.029, whose logit is -3.5.
# Recompute this prior for a clean corpus and override it without editing code.
OUTPUT_BIAS_INIT = float(os.environ.get("NOWCAST_OUTPUT_BIAS_INIT", "-3.5"))


class ConvLSTM3DCell(layers.Layer):
    """One 3D ConvLSTM step with layer-normalised gates and unit forget bias."""

    def __init__(self, filters, kernel_size, spatial_shape, padding='same',
                 activation='tanh', use_layer_norm=True, **kwargs):
        super(ConvLSTM3DCell, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.spatial_shape = spatial_shape  # (depth, height, width)
        self.padding = padding
        self.activation = tf.keras.activations.get(activation)
        self.use_layer_norm = use_layer_norm

        state_shape = self.spatial_shape + (self.filters,)
        self.state_size = (state_shape, state_shape)
        self.output_size = state_shape

    def build(self, input_shape):
        # No bias on the conv: biases are added explicitly after layer norm so
        # that normalisation cannot wash out the unit forget bias (FIX 1+4).
        self.conv = layers.Conv3D(
            filters=4 * self.filters,
            kernel_size=self.kernel_size,
            padding=self.padding,
            strides=(1, 1, 1),
            activation=None,
            use_bias=False,
        )

        if self.use_layer_norm:
            self.ln_i = layers.LayerNormalization(axis=-1, name='ln_i')
            self.ln_f = layers.LayerNormalization(axis=-1, name='ln_f')
            self.ln_o = layers.LayerNormalization(axis=-1, name='ln_o')
            self.ln_j = layers.LayerNormalization(axis=-1, name='ln_j')
            self.ln_c = layers.LayerNormalization(axis=-1, name='ln_c')

        # FIX 1: forget gate starts open (bias 1) so the cell state persists.
        self.bias_i = self.add_weight(shape=(self.filters,), initializer='zeros', name='bias_i')
        self.bias_f = self.add_weight(shape=(self.filters,), initializer='ones', name='bias_f')
        self.bias_o = self.add_weight(shape=(self.filters,), initializer='zeros', name='bias_o')
        self.bias_j = self.add_weight(shape=(self.filters,), initializer='zeros', name='bias_j')

        super(ConvLSTM3DCell, self).build(input_shape)

    def call(self, inputs, states):
        h_tm1, c_tm1 = states
        combined = tf.concat([inputs, h_tm1], axis=-1)
        z = self.conv(combined)
        z_i, z_f, z_o, z_j = tf.split(z, 4, axis=-1)

        if self.use_layer_norm:
            z_i, z_f, z_o, z_j = self.ln_i(z_i), self.ln_f(z_f), self.ln_o(z_o), self.ln_j(z_j)

        i = tf.sigmoid(z_i + self.bias_i)
        f = tf.sigmoid(z_f + self.bias_f)
        o = tf.sigmoid(z_o + self.bias_o)
        j = self.activation(z_j + self.bias_j)

        c = f * c_tm1 + i * j
        c_act = self.ln_c(c) if self.use_layer_norm else c
        h = o * self.activation(c_act)
        # The unnormalised cell state is carried forward (standard LN-LSTM).
        return h, [h, c]

    def get_config(self):
        config = super(ConvLSTM3DCell, self).get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "padding": self.padding,
            "activation": tf.keras.activations.serialize(self.activation),
            "spatial_shape": self.spatial_shape,
            "use_layer_norm": self.use_layer_norm,
        })
        return config


class ConvLSTM3D(layers.Layer):
    """Unrolls ConvLSTM3DCell over the time axis."""

    def __init__(self, filters, kernel_size, spatial_shape, padding='same',
                 activation='tanh', return_sequences=False, return_state=False,
                 use_layer_norm=True, **kwargs):
        super(ConvLSTM3D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.spatial_shape = spatial_shape
        self.use_layer_norm = use_layer_norm
        self.cell = ConvLSTM3DCell(filters, kernel_size, spatial_shape, padding,
                                   activation, use_layer_norm=use_layer_norm)

    def call(self, inputs, initial_state=None):
        # inputs: (batch, time, depth, height, width, channels)
        batch_size = tf.shape(inputs)[0]

        if initial_state is None:
            d, h, w = self.spatial_shape
            c = tf.zeros((batch_size, d, h, w, self.filters))
            h_state = tf.zeros((batch_size, d, h, w, self.filters))
            state = [h_state, c]
        else:
            state = initial_state

        outputs = []
        for inp in tf.unstack(inputs, axis=1):
            output, state = self.cell(inp, state)
            outputs.append(output)

        if self.return_sequences:
            return tf.stack(outputs, axis=1)
        if self.return_state:
            return outputs[-1], state[0], state[1]
        return outputs[-1]

    def get_config(self):
        config = super(ConvLSTM3D, self).get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "spatial_shape": self.spatial_shape,
            "padding": self.padding,
            "activation": self.activation,
            "return_sequences": self.return_sequences,
            "return_state": self.return_state,
            "use_layer_norm": self.use_layer_norm,
        })
        return config


def get_spatial_extractor():
    """Compress each (16,120,120) volume to (4,60,60) features."""
    return tf.keras.Sequential([
        layers.Conv3D(32, (3, 3, 3), padding='same', activation='relu'),
        layers.Conv3D(64, (3, 3, 3), strides=(2, 2, 2), padding='same', activation='relu'),
        layers.Conv3D(64, (3, 3, 3), padding='same', activation='relu'),
        layers.Conv3D(64, (3, 3, 3), strides=(2, 1, 1), padding='same', activation='relu'),
    ], name='spatial_extractor')


def get_decoder():
    """Expand (4,60,60) features back to a (16,120,120) reflectivity volume."""
    return tf.keras.Sequential([
        layers.Conv3DTranspose(64, (3, 3, 3), strides=(2, 2, 2), padding='same', activation='relu'),
        layers.Conv3D(64, (3, 3, 3), padding='same', activation='relu'),
        layers.Conv3DTranspose(64, (3, 3, 3), strides=(2, 1, 1), padding='same', activation='relu'),
        # Sigmoid keeps output in [0,1] like the targets. The configurable
        # conservative bias avoids a 0.5 start; recompute it from regenerated
        # training targets before a definitive experiment.
        layers.Conv3D(
            1, (1, 1, 1), padding='same', activation='sigmoid',
            bias_initializer=tf.keras.initializers.Constant(OUTPUT_BIAS_INIT),
        ),
    ], name='decoder')


def build_model(input_shape, input_length, output_length, use_skip=False):
    """Encoder-forecaster 3D-ConvLSTM.

    input_shape: (input_length, depth, height, width, channels)
    use_skip:    see module docstring — off by default (persistence risk).
    """
    inputs = layers.Input(shape=input_shape)

    spatial_extractor = get_spatial_extractor()
    decoder = get_decoder()

    encoded_frames = layers.TimeDistributed(spatial_extractor)(inputs)

    # 16x120x120 -> stride(2,2,2) -> 8x60x60 -> stride(2,1,1) -> 4x60x60
    spatial_shape = (4, 60, 60)

    # ---- Encoder ----
    x = ConvLSTM3D(64, (3, 3, 3), spatial_shape, return_sequences=True)(encoded_frames)
    x, h, c = ConvLSTM3D(64, (3, 3, 3), spatial_shape,
                         return_sequences=False, return_state=True)(x)

    # ---- Forecaster ----
    # Architectural adaptation: the published reference drives its forecaster
    # with zeros while information arrives through the recurrent state. Here we
    # repeat the final encoded frame as an additional input signal. This is not
    # a bug-for-bug correction and must be ablated against a faithful port.
    forecaster_input = layers.Lambda(
        lambda t: tf.repeat(t[:, -1:], output_length, axis=1),
        name='repeat_last_encoded',
    )(encoded_frames)

    x = ConvLSTM3D(64, (3, 3, 3), spatial_shape, return_sequences=True)(
        forecaster_input, initial_state=[h, c])
    x = ConvLSTM3D(64, (3, 3, 3), spatial_shape, return_sequences=True)(x)

    if use_skip:
        # Optional encoder->decoder skip (see module docstring for the caveat).
        x = layers.Concatenate(axis=-1)([x, forecaster_input])

    outputs = layers.TimeDistributed(decoder)(x)
    return Model(inputs, outputs)


if __name__ == "__main__":
    from nowcasting.config import INPUT_LENGTH
    from nowcasting.config import INPUT_CHANNELS
    from nowcasting.config import OUTPUT_LENGTH
    from nowcasting.config import TARGET_SHAPE

    model = build_model(
        (INPUT_LENGTH, *TARGET_SHAPE, INPUT_CHANNELS), INPUT_LENGTH, OUTPUT_LENGTH
    )
    model.summary()
