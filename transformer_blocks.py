from keras.saving import register_keras_serializable
import tensorflow as tf
from keras import layers

@register_keras_serializable()
class TransformerBlock(layers.Layer):

    def __init__(self, units, activation, num_heads, key_dim, value_dim, sequence_length, trainable=True, *args, **kwargs):
        super(TransformerBlock, self).__init__(*args, **kwargs, trainable=trainable)
        self.attention = layers.MultiHeadAttention(num_heads=num_heads,
                                                   key_dim=key_dim, value_dim=value_dim,
                                                   trainable=trainable)
        self.layer_norm1 = layers.LayerNormalization(trainable=trainable, epsilon=1e-07)
        self.fnn = layers.Dense(units, activation=activation, trainable=trainable)
        self.fnn2 = layers.Dense(value_dim, trainable=trainable)
        self.projection = layers.Dense(sequence_length, trainable=trainable)
        self.layer_norm2 = layers.LayerNormalization(trainable=trainable, epsilon=1e-07)

        self.units = units
        self.activation = activation
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.trainable = trainable
        self.sequence_length = sequence_length

    def call(self, query):
        attention_value = self.attention(query, query,
                                         attention_mask=tf.ones((query.shape[0] if query.shape[0] is not None else 1, self.num_heads,
                                                                     self.sequence_length, self.sequence_length)))
        layer_norm1 = self.layer_norm1(attention_value + query)
        fnn = self.fnn(layer_norm1)
        fnn2 = self.fnn2(fnn) + layer_norm1

        return fnn2

    def build(self, input_shape):
        super(TransformerBlock, self).build(input_shape)

    def get_config(self):
        config = {
            "units": self.units,
            "activation": self.activation,
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "trainable": self.trainable,
            "sequence_length": self.sequence_length,
        }

        base_config = super(TransformerBlock, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
