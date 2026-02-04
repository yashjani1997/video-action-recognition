# src/train.py
import tensorflow as tf
from src.dataset import load_dataset
from src.model import build_cnn_lstm_model

X_train, y_train, label2id, id2label = load_dataset(
    "data/train", "data/train.csv"
)

num_classes = len(label2id)

y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes)

model = build_cnn_lstm_model(num_classes)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train, y_train_oh,
    batch_size=2,
    epochs=5,
    validation_split=0.2
)

model.save("models/action_recognition_final.keras")
print("✅ Model training complete & saved")

