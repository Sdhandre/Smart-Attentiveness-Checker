import streamlit as st
import cv2
import mediapipe as mp
import joblib
import numpy as np
from PIL import Image
import tempfile

# Load trained model
clf = joblib.load("attention_model_2.pkl")

# Initialize Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

st.title("🎯 Smart Attentiveness Checker")
st.markdown("Upload an **image or video** to check attentiveness using Mediapipe + ML model.")

# File uploader
uploaded_file = st.file_uploader("Upload Image/Video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

if uploaded_file is not None:
    file_type = uploaded_file.type

    # Handle image files
    if "image" in file_type:
        img = Image.open(uploaded_file).convert("RGB")
        frame = np.array(img)
        rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = face_mesh.process(rgb)

        label = "No Face"
        color = (255, 255, 255)

        if results.multi_face_landmarks:
            row = []
            for lm in results.multi_face_landmarks[0].landmark:
                row += [lm.x, lm.y, lm.z]

            row = np.array(row).reshape(1, -1)
            pred = clf.predict(row)[0]

            if pred == 0:
                label = "ATTENTIVE"
                color = (0, 255, 0)
            else:
                label = "DROWSY"
                color = (0, 0, 255)

            cv2.putText(frame, label, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        st.image(frame, channels="RGB", caption=f"Prediction: {label}")

    # Handle video files
    elif "video" in file_type:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            label = "No Face"
            color = (255, 255, 255)

            if results.multi_face_landmarks:
                row = []
                for lm in results.multi_face_landmarks[0].landmark:
                    row += [lm.x, lm.y, lm.z]

                row = np.array(row).reshape(1, -1)
                pred = clf.predict(row)[0]

                if pred == 0:
                    label = "ATTENTIVE"
                    color = (0, 255, 0)
                else:
                    label = "DROWSY"
                    color = (0, 0, 255)

                cv2.putText(frame, label, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()
