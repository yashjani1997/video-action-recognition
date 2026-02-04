import os
import cv2
from src.inference import predict_video

# ---------------- CONFIG ----------------
video_path = "data/test/v_PlayingCello_g01_c01.mp4"
# --------------------------------------

print("Using video path:", video_path)

# ---- PREDICTION ----
label, confidence = predict_video(video_path)
print("Prediction:", label, "Confidence:", confidence)

# ---- VISUAL DEMO ----
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Video file open nahi ho rahi")
    exit()

window_name = "Action Recognition Demo"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    text = f"{label} ({confidence:.2f})"

    cv2.putText(
        frame,
        text,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(window_name, frame)

    # IMPORTANT: window ko alive rakhta hai
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
