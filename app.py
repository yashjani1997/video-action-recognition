import streamlit as st
import os

from src.video_utility import convert_to_mp4
from src.inference import predict_video
from src.hf_description_generator import generate_description

# Temp folder
os.makedirs("temp", exist_ok=True)

st.set_page_config(page_title="Video Action Recognition", layout="centered")
st.title("🎥 Video Action Recognition System")

uploaded_file = st.file_uploader(
    "Upload a video (any format)",
    type=["mp4", "avi", "mov", "mkv", "webm", "3gp"]
)

if uploaded_file:
    raw_path = f"temp/{uploaded_file.name}"
    mp4_path = "temp/converted.mp4"

    # Save uploaded file
    with open(raw_path, "wb") as f:
        f.write(uploaded_file.read())

    # Convert to MP4 if needed
    if not raw_path.lower().endswith(".mp4"):
        convert_to_mp4(raw_path, mp4_path)
        video_path = mp4_path
    else:
        video_path = raw_path

    st.video(video_path)

    with st.spinner("Analyzing video..."):
        label, confidence = predict_video(video_path)
        description = generate_description(label, confidence)

    st.success(f"🧠 Action: {label}")
    st.info(f"📊 Confidence: {confidence:.2f}")
    st.write(f"📝 Description: {description}")
