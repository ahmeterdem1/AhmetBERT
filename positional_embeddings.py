from keras.saving import register_keras_serializable
import tensorflow as tf
from keras import layers

@register_keras_serializable()
class PositionalEncoding(layers.Layer):
    def __init__(self, max_position, d_model, *args, **kwargs):
        super(PositionalEncoding, self).__init__(*args, **kwargs)
        self.max_position = max_position
        self.d_model = d_model

    def get_angles(self, position, i):
        angles = 1 / tf.pow(tf.cast(10000, tf.float32),
                            tf.cast(2 * (i // 2), tf.float32) / tf.cast(self.d_model, tf.float32))
        return tf.cast(position, tf.float32) * angles

    def positional_encoding(self, position):
        angle_rads = self.get_angles(tf.range(position)[:, tf.newaxis],
                                     tf.range(self.d_model)[tf.newaxis, :])

        angle_rads = tf.where(tf.math.equal(tf.math.floormod(angle_rads, 2), 0),
                              tf.math.sin(angle_rads),
                              tf.math.cos(angle_rads))

        pos_encoding = angle_rads[tf.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        seq_length = tf.shape(inputs)[1]
        pos_encoding = self.positional_encoding(seq_length)
        return inputs + pos_encoding[:, :seq_length, :]

    def get_config(self):
        config = {
            "max_position": self.max_position,
            "d_model": self.d_model,
            "trainable": self.trainable,
        }

        base_config = super(PositionalEncoding, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
