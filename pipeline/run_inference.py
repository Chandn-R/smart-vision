import sys
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import deque

# --- PROJECT SETUP ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(ROOT_DIR)
sys.path.append(ROOT_DIR)

# Import your existing modules
from src.modeling.detectors.yolo_detector import YOLODetector
from src.modeling.trackers.byte_tracker import ByteTracker
from src.modeling.pose_estimators.mediapipe import MediaPipeEstimator

# --- CONFIGURATION ---
YOLO_MODEL_PATH = "../models/yolov11s_fine_tune.pt"
LSTM_MODEL_PATH = "../models/lstm_action_recognition_pro.pth" # <--- New LSTM Path
WEBCAM_ID = 0
CONFIDENCE_THRESHOLD = 0.45
SEQUENCE_LENGTH = 30  # Must match training

# Your Custom Classes
CLASS_COLORS = {
    0: (0, 255, 0),    # Person: Green
    1: (0, 0, 255),    # Gun: Red
    2: (0, 0, 255),    # Long Gun: Red
    3: (0, 0, 255),    # Knife: Red
    4: (0, 165, 255),  # Blunt Weapon: Orange
    5: (0, 165, 255),  # Burglary Tool: Orange
}
PERSON_CLASS_ID = 0
WEAPON_CLASS_IDS = [1, 2, 3, 4, 5]
GUN_CLASS_IDS = [1, 2] # Specific IDs for firearms

# --- 1. DEFINE LSTM MODEL (Must match Training) ---
class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BidirectionalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) 
        return out

def get_color(class_id):
    return CLASS_COLORS.get(class_id, (255, 255, 255))

def run_pipeline():
    # --- A. INITIALIZE MODELS ---
    print("🚀 Initializing SmartVision Pipeline...")
    
    # 1. Load YOLO
    detector = YOLODetector()
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"❌ Error: YOLO model not found at {YOLO_MODEL_PATH}")
        return
    detector.load_model(YOLO_MODEL_PATH)
    
    # 2. Load LSTM
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    
    print(f"   Loading LSTM Brain from {LSTM_MODEL_PATH}...")
    if not os.path.exists(LSTM_MODEL_PATH):
        print(f"❌ Error: LSTM model not found at {LSTM_MODEL_PATH}")
        return

    checkpoint = torch.load(LSTM_MODEL_PATH, map_location=device)
    lstm_model = BidirectionalLSTM(
        checkpoint['input_size'], 
        checkpoint['hidden_size'], 
        checkpoint['num_layers'], 
        len(checkpoint['class_map'])
    ).to(device)
    lstm_model.load_state_dict(checkpoint['model_state_dict'])
    lstm_model.eval()
    
    class_map = checkpoint['class_map'] # {'normal': 0, 'violence': 1...}
    inv_class_map = {v: k for k, v in class_map.items()} # {0: 'normal'...}

    # 3. Helpers
    tracker = ByteTracker()
    pose_estimator = MediaPipeEstimator()
    cap = cv2.VideoCapture(WEBCAM_ID)
    
    # HISTORY STORAGE: { track_id: deque([frame1, frame2...]) }
    person_history = {} 

    if not cap.isOpened():
        print(f"❌ Error: Cannot open webcam {WEBCAM_ID}")
        return

    print("✅ System Live. Press 'Q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        h_img, w_img, _ = frame.shape
        results = detector.predict(frame, conf=CONFIDENCE_THRESHOLD)
        
        detections_for_tracking = []
        visible_weapons = [] # List of weapon class IDs found in this frame

        # --- B. DETECTION LOOP ---
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = detector.class_names.get(cls_id, "Unknown")

                # Weapon Logic
                if cls_id in WEAPON_CLASS_IDS:
                    visible_weapons.append(cls_id)
                    color = get_color(cls_id)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Alert Label
                    label = f"THREAT: {class_name}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + tw, y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Person Logic
                elif cls_id == PERSON_CLASS_ID:
                    detections_for_tracking.append(([x1, y1, x2, y2], conf, cls_id))

        # --- C. TRACKING & LSTM LOOP ---
        tracked_objects = tracker.update(detections_for_tracking)

        if len(tracked_objects) > 0:
            for i in range(len(tracked_objects)):
                x1, y1, x2, y2 = map(int, tracked_objects.xyxy[i])
                track_id = int(tracked_objects.tracker_id[i])

                # 1. Manage History
                if track_id not in person_history:
                    person_history[track_id] = deque(maxlen=SEQUENCE_LENGTH)

                # 2. Crop Person
                x1_c, y1_c = max(0, x1), max(0, y1)
                x2_c, y2_c = min(w_img, x2), min(h_img, y2)
                person_crop = frame[y1_c:y2_c, x1_c:x2_c]

                landmarks_list = None
                
                # 3. Pose Estimation
                if person_crop.size > 0:
                    # Note: landmarks_list is the flat [x,y,z,v...] list
                    landmarks_list, landmarks_obj = pose_estimator.predict(person_crop)

                    if landmarks_obj:
                        pose_estimator.visualize(person_crop, landmarks_obj)
                        frame[y1_c:y2_c, x1_c:x2_c] = person_crop # Paste skeleton back
                
                # 4. Update Sequence
                if landmarks_list:
                    person_history[track_id].append(landmarks_list)
                else:
                    # Pad with zeros if pose failed this frame
                    if len(person_history[track_id]) > 0:
                        person_history[track_id].append([0.0] * 132)

                # 5. LSTM Inference (Only if we have 30 frames)
                action_label = "Scanning..."
                action_prob = 0.0
                threat_level = "SAFE"
                box_color = (0, 255, 0) # Green default

                if len(person_history[track_id]) == SEQUENCE_LENGTH:
                    seq = np.array(person_history[track_id], dtype=np.float32)
                    seq_tensor = torch.tensor(seq).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        output = lstm_model(seq_tensor)
                        probs = torch.softmax(output, dim=1)
                        top_p, top_class = torch.topk(probs, 1)
                        
                        if top_p.item() > 0.6: # Confidence Threshold
                            action_label = inv_class_map[top_class.item()]
                            action_prob = top_p.item()

                    # --- D. LOGIC MATRIX (FUSION) ---
                    # Rule 1: Gun + Shooting = ACTIVE SHOOTER
                    has_gun = any(w in GUN_CLASS_IDS for w in visible_weapons)
                    has_weapon = len(visible_weapons) > 0
                    
                    if action_label == "shooting" and has_gun:
                        threat_level = "CRITICAL: SHOOTER"
                        box_color = (0, 0, 255) # Red
                    
                    # Rule 2: Knife + Violence = STABBING/ATTACK
                    elif action_label == "violence" and (3 in visible_weapons):
                        threat_level = "CRITICAL: KNIFE ATTACK"
                        box_color = (0, 0, 255) # Red

                    # Rule 3: Violence (No Weapon) = FIGHTING
                    elif action_label == "violence":
                        threat_level = "HIGH: FIGHTING"
                        box_color = (0, 165, 255) # Orange
                    
                    # Rule 4: Shooting Stance (No Gun) = SUSPICIOUS
                    elif action_label == "shooting":
                        threat_level = "WARN: SUSPICIOUS STANCE"
                        box_color = (0, 255, 255) # Yellow

                # 6. Draw UI
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                
                # Top Label (ID + Action)
                info_text = f"ID:{track_id} {action_label}"
                if action_prob > 0: info_text += f" ({action_prob:.0%})"
                cv2.putText(frame, info_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                
                # Bottom Label (Threat Level - Only if unsafe)
                if threat_level != "SAFE":
                    cv2.putText(frame, threat_level, (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

        cv2.imshow("SmartVision Pro - Threat Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pipeline()