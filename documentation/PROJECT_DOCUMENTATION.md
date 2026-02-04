
# Video Action Recognition using CNN + LSTM

## Project Overview
This project implements an end-to-end **Video Action Recognition system** using a hybrid **CNN + LSTM** deep learning architecture on a subset of the UCF-101 dataset.

The system is capable of:
- Loading raw video files
- Extracting frames
- Learning spatial + temporal features
- Predicting human actions
- Displaying predictions on video playback

---

## Dataset
- Dataset: UCF-101 (subset)
- Classes:
  - CricketShot
  - PlayingCello
  - Punch
  - ShavingBeard
  - TennisSwing

CSV files are used to map video filenames to class labels.

---

## Project Structure
```
Video Action Recognition/
├── data/
│   ├── train/
│   ├── test/
│   ├── train.csv
│   └── test.csv
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── inference.py
├── notebooks/
│   ├── dataset_explore.ipynb
│   ├── model_experiment.ipynb
│   └── training_analysis.ipynb
├── visual_test.py
└── saved_model.keras
```

---

## Data Processing
- Fixed number of frames extracted per video
- Frames resized and normalized
- Short videos padded

Output shape:
(samples, frames, height, width, channels)

---

## Model Architecture
- CNN: MobileNetV2 (ImageNet pretrained)
- TimeDistributed wrapper
- LSTM for temporal learning
- Dense Softmax output layer

---

## Training
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Validation split used
- Model saved in `.keras` format

---

## Evaluation
- Confusion Matrix
- Precision, Recall, F1-score
- ~87% accuracy achieved

---

## Inference & Demo
- Load trained model
- Predict action from unseen video
- Overlay prediction on video playback

---

## Outcome
A complete, production-style video action recognition pipeline.
