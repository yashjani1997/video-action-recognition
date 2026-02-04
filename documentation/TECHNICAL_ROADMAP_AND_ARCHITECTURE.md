
# Technical Roadmap & Architecture

## Phase-A: Core System (Completed)
1. Dataset exploration
2. Frame extraction pipeline
3. CNN + LSTM model design
4. Training and evaluation
5. Local visual demo

---

## Why CNN + LSTM?
- CNN: Extracts spatial features from frames
- LSTM: Learns temporal motion patterns
- Combined approach handles video data effectively

---

## Architecture Flow
Video → Frame Sampling → CNN → Feature Vectors → LSTM → Softmax → Action Label

---

## Key Engineering Decisions
- Transfer learning for faster convergence
- Fixed frame sampling for efficiency
- Modular code structure for scalability

---

## Future Enhancements
- Streamlit web deployment
- Real-time webcam inference
- Attention mechanisms
- Transformer-based video models
- Model optimization for edge devices

---

## Applications
- Surveillance & security
- Sports analytics
- Human activity monitoring
- Smart video indexing

---

## Engineer's Note
This project follows an industry-style ML workflow from data to deployment.
