# 🎥 Video Action Recognition using CNN + LSTM

An end-to-end **Video Action Recognition system** built using a hybrid **CNN + LSTM** deep learning architecture on a subset of the UCF-101 dataset.

---

## 🚀 Project Overview

This system is capable of:
- Loading raw video files
- Extracting and preprocessing frames
- Learning **spatial features** via CNN (MobileNetV2)
- Learning **temporal patterns** via LSTM
- Predicting human actions with confidence scores
- Displaying live predictions as overlay on video playback

---

## 📂 Dataset

- **Dataset:** UCF-101 (subset)
- **Classes:** CricketShot, PlayingCello, Punch, ShavingBeard, TennisSwing
- CSV files used to map video filenames to class labels
- Raw video files not included due to size constraints

---

## 📁 Project Structure

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
│   └── inference.py
├── notebooks/
│   ├── dataset_explore.ipynb
│   ├── model_experiment.ipynb
│   └── training_analysis.ipynb
├── visual_test.py
└── saved_model.keras
```

---

## ⚙️ Data Processing

- **16 frames** extracted per video (fixed)
- Frames resized to **112 x 112**
- Frames normalized (pixel values scaled)
- Short videos padded to maintain fixed length
- Output shape: `(samples, 16, 112, 112, 3)`

---

## 🧠 Model Architecture

```
Video → Frames (16) → CNN (MobileNetV2) → LSTM → Dropout → Softmax → Action Label
```

| Layer | Output Shape | Parameters |
|---|---|---|
| Input | (None, 16, 112, 112, 3) | 0 |
| TimeDistributed (MobileNetV2) | (None, 16, 256) | 2,585,920 |
| LSTM (128 units) | (None, 128) | 197,120 |
| Dropout | (None, 128) | 0 |
| Dense Softmax (5 classes) | (None, 5) | 645 |

**Total Parameters:** 3,835,089 (14.63 MB)  
**Trainable Parameters:** 525,701 (2.01 MB)  
**Non-trainable (frozen CNN):** 2,257,984 (8.61 MB)

**Why MobileNetV2?**  
Lightweight pretrained CNN — strong spatial features with minimal compute cost, ideal for frame-level feature extraction in video tasks.

---

## 🏋️ Training

- **Optimizer:** Adam
- **Loss:** Categorical Crossentropy
- **Validation split** used during training
- Model saved in `.keras` format

---

## 📊 Results

**Overall Accuracy: ~87%**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| CricketShot | 0.71 | 0.82 | 0.76 | 49 |
| PlayingCello | 0.91 | 0.95 | 0.93 | 44 |
| Punch | 0.88 | 0.97 | 0.93 | 39 |
| ShavingBeard | 1.00 | 0.91 | 0.95 | 43 |
| TennisSwing | 0.88 | 0.71 | 0.79 | 49 |
| **Weighted Avg** | **0.87** | **0.87** | **0.87** | **224** |

- **ShavingBeard** achieved perfect precision (1.00)
- **Punch** achieved highest recall (0.97)
- **CricketShot** & **TennisSwing** are harder to distinguish (motion overlap)

---

## 🔍 Inference & Demo

- Load trained model (`saved_model.keras`)
- Predict action from unseen video
- Overlay predicted action label + confidence on video playback (OpenCV)

▶️ **[Watch Demo](https://drive.google.com/file/d/14CyVO7UjqGsABzvGAZo046OA9o8Um4p8/view?usp=sharing)**

---

## ⚠️ Limitations

- Trained on **5 classes only** (UCF-101 has 101 classes)
- Small subset — performance may vary on unseen action types
- Fixed 16-frame window may miss context in longer action sequences

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow / Keras |
| CNN Backbone | MobileNetV2 (ImageNet pretrained) |
| Video Processing | OpenCV |
| Data | NumPy, Pandas |

---

## ⚙️ How to Run Locally

```bash
# Clone the repository
git clone https://github.com/yashjani1997/video-action-recognition.git
cd video-action-recognition

# Install dependencies
pip install -r requirements.txt

# Run inference on a video
python visual_test.py
```

---

## 🧠 Key Learnings

- End-to-end video ML pipeline design
- Transfer learning with TimeDistributed CNN
- Temporal modeling with LSTM
- Handling variable-length video with fixed frame sampling
- Production-style project structure (src/, notebooks/, inference)

---

## 👤 Author

**Yash Jani**  
Data Analyst & Machine Learning Enthusiast  
[GitHub: yashjani1997](https://github.com/yashjani1997)
