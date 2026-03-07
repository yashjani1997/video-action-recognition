# src/inference.py
import numpy as np
import tensorflow as tf
from src.dataset import extract_frames
import os

MODEL_PATH = "models/action_recognition_final.h5"

ID2LABEL = {
    0: "CricketShot",
    1: "PlayingCello",
    2: "Punch",
    3: "ShavingBeard",
    4: "TennisSwing"
}

_model = None

# ✅ Patch for Keras 2 vs Keras 3 'batch_shape' mismatch
class FixedInputLayer(tf.keras.layers.InputLayer):
    def __init__(self, **kwargs):
        if 'batch_shape' in kwargs:
            kwargs['shape'] = kwargs.pop('batch_shape')[1:]
        super().__init__(**kwargs)

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        _model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            custom_objects={'InputLayer': FixedInputLayer}
        )
    return _model


def predict_video(video_path):
    model = get_model()
    frames = extract_frames(video_path)
    frames = np.expand_dims(frames, axis=0)
    preds = model.predict(frames)
    class_id = np.argmax(preds)
    return ID2LABEL[class_id], float(np.max(preds))
