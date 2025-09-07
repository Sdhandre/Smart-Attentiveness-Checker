import streamlit as st
import cv2
import mediapipe as mp
import joblib
import numpy as np
from PIL import Image

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

st.title("🎯 Real-Time Attention Detection")
st.markdown("Webcam-based attention monitoring using Mediapipe FaceMesh + ML model.")

# Camera input
camera_input = st.camera_input("📷 Turn on your camera")

if camera_input:
    # Convert to CV2 image
    image = Image.open(camera_input)
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

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

    # Show processed frame
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
