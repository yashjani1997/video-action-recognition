# save_weights.py
import tensorflow as tf
import numpy as np
import pickle

print("Loading old model...")
old_model = tf.keras.models.load_model(
    "models/action_recognition_final.h5",
    compile=False
)

weights = old_model.get_weights()
print(f"Total weight arrays: {len(weights)}")

# pickle se save karo — numpy inhomogeneous shapes handle nahi karta
with open("models/model_weights.pkl", "wb") as f:
    pickle.dump(weights, f)

print("Done! models/model_weights.pkl saved!")
