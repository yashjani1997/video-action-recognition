import tensorflow as tf
print("TF version:", tf.__version__)

model = tf.keras.models.load_model(
    "models/action_recognition_final.keras", 
    compile=False
)
model.save("models/action_recognition_final.h5")
print("Done!")