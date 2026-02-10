import cv2
import os

input_video = "data/test/v_PlayingCello_g01_c03.avi"
output_video = "data/test/v_PlayingCello_g01_c03.mp4"

cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print("❌ Input AVI open nahi ho rahi")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("FPS:", fps, "Size:", width, height)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    frame_count += 1

cap.release()
out.release()

print(f"✅ Conversion complete: {output_video}")
print("Total frames:", frame_count)
