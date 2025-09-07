import streamlit as st
import cv2
import mediapipe as mp
import joblib
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Load trained model
clf = joblib.load("attention_model_2.pkl")

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

st.title("🎯 Real-Time Attention Detection")
st.markdown("Live webcam-based attention monitoring using Mediapipe FaceMesh + ML model.")

# WebRTC Config
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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

            cv2.putText(img, label, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return img

# Start WebRTC streamer
webrtc_streamer(
    key="attention-detection",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False}
)
