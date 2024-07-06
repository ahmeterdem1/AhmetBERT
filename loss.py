from keras.saving import register_keras_serializable
import tensorflow as tf
from keras import losses

@register_keras_serializable()
def masked_sparse_categorical_crossentropy(y_true, y_pred):
    mask = tf.not_equal(y_true, -100)
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    loss = losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)
    return tf.reduce_mean(loss)
