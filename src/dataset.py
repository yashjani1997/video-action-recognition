# src/dataset.py
import os
import cv2
import numpy as np
import pandas as pd

IMG_SIZE = 112
NUM_FRAMES = 16


def load_csv(csv_path):
    df = pd.read_csv(csv_path)

    classes = sorted(df["tag"].unique())
    label2id = {label: i for i, label in enumerate(classes)}
    id2label = {i: label for label, i in label2id.items()}

    df["label_id"] = df["tag"].map(label2id)
    return df, label2id, id2label


def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idxs = np.linspace(0, total_frames - 1, NUM_FRAMES).astype(int)
    frames = []

    idx_set = set(frame_idxs)
    idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if idx in idx_set:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = frame.astype("float32") / 255.0
            frames.append(frame)

        idx += 1

    cap.release()

    while len(frames) < NUM_FRAMES:
        frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32))

    return np.array(frames)


def load_dataset(video_dir, csv_path):
    df, label2id, id2label = load_csv(csv_path)

    X, y = [], []

    for _, row in df.iterrows():
        video_path = os.path.join(video_dir, row["video_name"])
        if not os.path.exists(video_path):
            continue

        X.append(extract_frames(video_path))
        y.append(row["label_id"])

    return np.array(X), np.array(y), label2id, id2label
