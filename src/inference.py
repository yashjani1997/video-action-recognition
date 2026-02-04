# src/inference.py
import numpy as np
import tensorflow as tf
from src.dataset import extract_frames

MODEL_PATH = "models/action_recognition_final.keras"

model = tf.keras.models.load_model(MODEL_PATH)

ID2LABEL = {
    0: "CricketShot",
    1: "PlayingCello",
    2: "Punch",
    3: "ShavingBeard",
    4: "TennisSwing"
}


def predict_video(video_path):
    frames = extract_frames(video_path)
    frames = np.expand_dims(frames, axis=0)

    preds = model.predict(frames)
    class_id = np.argmax(preds)

    return ID2LABEL[class_id], float(np.max(preds))
