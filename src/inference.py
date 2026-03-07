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

# ✅ Lazy loading — model sirf tab load hoga jab predict_video call ho
_model = None

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"❌ Model file nahi mila: '{MODEL_PATH}'\n"
                f"Apne GitHub repo mein 'models/' folder mein "
                f"'action_recognition_final.keras' file daalo."
            )
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return _model


def predict_video(video_path):
    model = get_model()
    frames = extract_frames(video_path)
    frames = np.expand_dims(frames, axis=0)

    preds = model.predict(frames)
    class_id = np.argmax(preds)

    return ID2LABEL[class_id], float(np.max(preds))
