# Video Action Recognition using CNN + LSTM

This project implements an end-to-end **video action recognition system**
using a hybrid **CNN + LSTM** deep learning architecture on a subset of the
UCF-101 dataset.

## 🚀 Features
- Video frame extraction & preprocessing
- Pretrained CNN (MobileNetV2) for spatial features
- Custom LSTM for temporal modeling
- Training, evaluation & inference pipeline
- Live video prediction with visual overlay (OpenCV)

## 🧠 Model Architecture
Video → Frames → CNN (MobileNetV2) → LSTM → Softmax → Action Label

## 📊 Dataset
UCF-101 (subset) with the following classes:
- CricketShot
- PlayingCello
- Punch
- ShavingBeard
- TennisSwing

> Note: Raw video files are not included due to size constraints.

## 🛠 Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- NumPy, Pandas

## ▶️ Demo
Local visual demo overlays predicted action and confidence on video playback.
