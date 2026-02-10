import cv2
from src.inference import predict_video
from src.hf_description_generator import generate_description

video_path = "data/test/v_PlayingCello_g01_c03.mp4"

print("Using video path:", video_path)

# ---- STEP 1: Prediction ----
label, confidence = predict_video(video_path)

print("Prediction:", label)
print("Confidence:", confidence)

# ---- STEP 2: Open video FIRST ----
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Video file open nahi ho rahi")
    exit()

# ---- STEP 3: Generate description (safe) ----
description = generate_description(label, confidence)
print("Description:", description)

window_name = "Video Action Recognition Demo"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.putText(frame, f"Action: {label} ({confidence:.2f})",
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.putText(frame, f"Desc: {description}",
                (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow(window_name, frame)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
