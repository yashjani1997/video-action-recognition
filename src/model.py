# src/model.py
import tensorflow as tf


def build_cnn_lstm_model(num_classes):
    base_cnn = tf.keras.applications.MobileNetV2(
        input_shape=(112, 112, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg"
    )

    base_cnn.trainable = False

    inputs = tf.keras.Input(shape=(16, 112, 112, 3))
    x = tf.keras.layers.TimeDistributed(base_cnn)(inputs)
    x = tf.keras.layers.LSTM(128)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs)
