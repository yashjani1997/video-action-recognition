# rebuild_model.py
# E:\video-action-recognition\ mein save karo aur run karo

import tensorflow as tf
import numpy as np

print("TF:", tf.__version__)
print("Keras:", tf.keras.__version__)

# Step 1: Purana model load karo
old_model = tf.keras.models.load_model(
    "models/action_recognition_final.h5",
    compile=False
)

print("Old model loaded!")
old_model.summary()

# Step 2: Naya model same architecture se banao
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
outputs = tf.keras.layers.Dense(5, activation="softmax")(x)

new_model = tf.keras.Model(inputs, outputs)

# Step 3: Purane weights copy karo
new_model.set_weights(old_model.get_weights())
print("Weights copied!")

# Step 4: Save karo
new_model.save("models/action_recognition_rebuilt.h5")
print("Done! models/action_recognition_rebuilt.h5 ban gaya")
