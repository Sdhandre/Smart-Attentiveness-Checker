import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import joblib
from ultralytics import YOLO
from collections import defaultdict
import math

# --- 1. INITIALIZATION (No changes here) ---

# Load the drowsiness detection model and feature names
try:
    clf = joblib.load("attention_model_3.pkl")
    training_data = pd.read_csv("face_landmarks_dataset.csv")
    feature_names = training_data.drop("label", axis=1).columns.tolist()
except FileNotFoundError:
    print("Error: Model or dataset file not found.")
    print("Please make sure 'attention_model_3.pkl' and 'face_landmarks_dataset.csv' are in the correct directory.")
    exit()

# Load the YOLOv8 model for object detection
model = YOLO('yolo11n.pt')  # 'n' is for the nano version, fast and light

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=6,  # Handle up to 6 people as requested
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- 2. MULTI-PERSON TRACKER (No changes here) ---

class SimpleCentroidTracker:
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = {}  # Stores ID -> centroid
        self.disappeared = {}  # Stores ID -> frames it has been disappeared
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (startX, startY, endX, endY) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = np.zeros((len(input_centroids), len(object_centroids)))
            for i in range(len(input_centroids)):
                for j in range(len(object_centroids)):
                    dist = math.sqrt((input_centroids[i][0] - object_centroids[j][0])**2 + (input_centroids[i][1] - object_centroids[j][1])**2)
                    D[i, j] = dist
            
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows, used_cols = set(), set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                object_id = object_ids[col]
                self.objects[object_id] = input_centroids[row]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, len(input_centroids))).difference(used_rows)
            unused_cols = set(range(0, len(object_centroids))).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    self.register(input_centroids[row])
            else:
                for col in unused_cols:
                    object_id = object_ids[col]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
        
        return self.objects

# --- 3. MAIN APPLICATION SETUP ---

tracker = SimpleCentroidTracker()

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Data logging for summary
session_log = defaultdict(list)
frame_count = 0

# --- OPTIMIZATION VARIABLES ---
detection_interval = 1  # Run detection every 3 frames
processing_width = 900  # Resize frames to this width for faster processing
last_known_results = {} # Stores the latest results for each person ID

# --- 4. MAIN APPLICATION LOOP ---

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        
        # --- PERIODIC DETECTION (Runs every 'detection_interval' frames) ---
        if frame_count % detection_interval == 0:
            # 1. RESIZE FRAME FOR PERFORMANCE
            original_H, original_W, _ = frame.shape
            scale_factor = original_W / processing_width
            processing_height = int(original_H / scale_factor)
            small_frame = cv2.resize(frame, (processing_width, processing_height))
            
            # 2. RUN MODELS ON THE SMALLER FRAME
            # --- Object Detection (Phones) ---
            yolo_results = model(small_frame, verbose=False)
            phone_boxes = []
            for result in yolo_results:
                for box in result.boxes:
                    if int(box.cls) == 67: # Class for 'cell phone'
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        phone_boxes.append((x1, y1, x2, y2))
            
            # --- Face Detection and Landmark Extraction ---
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            results_mesh = face_mesh.process(rgb_frame)
            
            face_rects, face_landmarks_list = [], []
            if results_mesh.multi_face_landmarks:
                for face_landmarks in results_mesh.multi_face_landmarks:
                    h, w, c = small_frame.shape
                    cx_min, cy_min = w, h
                    cx_max, cy_max = 0, 0
                    for lm in face_landmarks.landmark:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cx_min, cy_min = min(cx_min, cx), min(cy_min, cy)
                        cx_max, cy_max = max(cx_max, cx), max(cy_max, cy)
                    
                    face_rects.append((cx_min, cy_min, cx_max, cy_max))
                    face_landmarks_list.append(face_landmarks)
            
            # 3. UPDATE TRACKER AND PROCESS EACH PERSON
            tracked_faces = tracker.update(face_rects)
            
            # Clear previous results and repopulate with new ones
            last_known_results.clear()

            for i, (object_id, centroid) in enumerate(tracked_faces.items()):
                if i < len(face_landmarks_list):
                    face_landmarks = face_landmarks_list[i]
                    face_box = face_rects[i]
                    
                    # Predict Drowsiness
                    row = [coord for lm in face_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]
                    X_live = pd.DataFrame([row], columns=feature_names)
                    drowsy_pred = clf.predict(X_live)[0]

                    # Check for phone usage by proximity
                    is_using_phone = False
                    for phone_box in phone_boxes:
                        dist_x = max(0, face_box[0] - phone_box[2], phone_box[0] - face_box[2])
                        dist_y = max(0, face_box[1] - phone_box[3], phone_box[1] - face_box[3])
                        if dist_x < 50 and dist_y < 50:
                            is_using_phone = True
                            break
                    
                    # Decision Logic
                    if is_using_phone:
                        status, color = "USING PHONE", (255, 165, 0)
                    elif drowsy_pred == 1:
                        status, color = "DROWSY", (0, 0, 255)
                    else:
                        status, color = "ATTENTIVE", (0, 255, 0)
                    
                    # Log and store the results for drawing
                    session_log[object_id].append(status)
                    last_known_results[object_id] = (face_box, status, color)

        # --- DRAWING (Runs on every frame for a smooth display) ---
        for object_id, (box, status, color) in last_known_results.items():
            # Scale coordinates from the small frame to the original frame
            x1, y1, x2, y2 = [int(coord * scale_factor) for coord in box]
            
            # Draw phone boxes detected in the last processed frame
            for p_box in phone_boxes:
                px1, py1, px2, py2 = [int(c * scale_factor) for c in p_box]
                cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 165, 0), 2)
                cv2.putText(frame, 'Phone', (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)

            # Draw face box and status
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"Person {object_id}: {status}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
        cv2.imshow('Student Attentiveness Monitor', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # --- 5. GENERATE SUMMARY (No changes here) ---
    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "="*50)
    print("           SESSION SUMMARY")
    print("="*50)

    if not session_log:
        print("No activity was logged.")
    else:
        for person_id, states in session_log.items():
            total_frames = len(states)
            if total_frames == 0: continue
            
            counts = defaultdict(int)
            for state in states:
                counts[state] += 1
            
            attentive_p = (counts["ATTENTIVE"] / total_frames) * 100
            drowsy_p = (counts["DROWSY"] / total_frames) * 100
            phone_p = (counts["USING PHONE"] / total_frames) * 100
            
            print(f"\n--- Person ID: {person_id} ---")
            print(f"  Total Time in View: {total_frames * detection_interval / 20:.1f} seconds (approx @ 20 FPS)")
            print(f"  - Attentive: {attentive_p:.1f}%")
            print(f"  - Drowsy:    {drowsy_p:.1f}%")
            print(f"  - Using Phone: {phone_p:.1f}%")
    
    print("\n" + "="*50)
