# src/inference.py
import numpy as np
import tensorflow as tf
from src.dataset import extract_frames
import os
import pickle

WEIGHTS_PATH = "models/model_weights.pkl"

ID2LABEL = {
    0: "CricketShot",
    1: "PlayingCello",
    2: "Punch",
    3: "ShavingBeard",
    4: "TennisSwing"
}

_model = None

def build_model():
    base_cnn = tf.keras.applications.MobileNetV2(
        input_shape=(112, 112, 3),
        include_top=False,
        weights=None,
        pooling="avg"
    )
    base_cnn.trainable = False
    inputs = tf.keras.Input(shape=(16, 112, 112, 3))
    x = tf.keras.layers.TimeDistributed(base_cnn)(inputs)
    x = tf.keras.layers.LSTM(128)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(5, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(WEIGHTS_PATH):
            raise FileNotFoundError(f"Weights not found: {WEIGHTS_PATH}")

        model = build_model()

        # Dummy pass to build all layers
        dummy = np.zeros((1, 16, 112, 112, 3), dtype=np.float32)
        model(dummy, training=False)

        # Load weights from pickle
        with open(WEIGHTS_PATH, "rb") as f:
            weights = pickle.load(f)

        model.set_weights(weights)
        _model = model
        print("✅ Model loaded!")
    return _model

def predict_video(video_path):
    model = get_model()
    frames = extract_frames(video_path)
    frames = np.expand_dims(frames, axis=0)
    preds = model.predict(frames)
    class_id = np.argmax(preds)
    return ID2LABEL[class_id], float(np.max(preds))
