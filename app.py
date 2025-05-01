import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import tempfile
import base64

# Load the trained deepfake detection model
model = tf.keras.models.load_model("model/deepfake_model.keras")

# Function to extract multiple frames from the video
def extract_frames(video_path, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // num_frames)  # Ensure spacing

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)
        success, frame = cap.read()
        if success:
            frame = cv2.resize(frame, (128, 128))  # Resize to model's expected input
            frames.append(frame)
    
    cap.release()

    if len(frames) < num_frames:
        frames += [frames[-1]] * (num_frames - len(frames))  # Repeat last frame if not enough
    
    frames = np.array(frames, dtype=np.float32) / 255.0  # Normalize
    frames = np.expand_dims(frames, axis=0)  # Add batch dimension (1, 10, 128, 128, 3)
    return frames

# Function to set background image in Streamlit
def set_background(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    page_bg = f'''
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    '''
    st.markdown(page_bg, unsafe_allow_html=True)

# Set background
set_background("deep_fake.jpg")

# Web App Title
st.markdown("<h1 style='text-align: center; color: white;'>Deepfake Detection</h1>", unsafe_allow_html=True)

# File Upload Section
uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file)

    # Save the uploaded video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        video_path = temp_video.name

    # Extract multiple frames from the video
    frames = extract_frames(video_path)
    
    # Make a prediction
    prediction = model.predict(frames)
    predicted_label = np.argmax(prediction, axis=1)[0]  # Get the highest probability class

    # Map the prediction to real/fake
    result = "Fake Video ❌" if predicted_label == 1 else "Real Video ✅"

    # Display the result
    st.markdown(f"<div style='padding: 10px; background-color: black; border-radius: 10px; text-align: center;'>"
                f"<h2 style='color: white;'>{result}</h2></div>", unsafe_allow_html=True)
