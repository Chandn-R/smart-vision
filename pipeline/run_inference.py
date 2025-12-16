import sys
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import logging
import time

# --- PROJECT SETUP ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(ROOT_DIR)
sys.path.append(ROOT_DIR)

# Import your existing modules
from src.modeling.detectors.yolo_detector import YOLODetector
from src.modeling.trackers.byte_tracker import ByteTracker
from src.modeling.pose_estimators.mediapipe import MediaPipeEstimator

# --- CONFIGURATION ---
YOLO_MODEL_PATH = os.path.join(ROOT_DIR, "models", "yolov11s_fine_tune.pt")
LSTM_MODEL_PATH = os.path.join(ROOT_DIR, "models", "lstm_action_recognition_pro.pth")
WEBCAM_ID = 0
CONFIDENCE_THRESHOLD = 0.45
SEQUENCE_LENGTH = 30  # Must match training

# Input/Output Directories
INPUT_VIDEO_DIR = os.path.join(ROOT_DIR, "data", "input_videos")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
LOG_DIR = os.path.join(ROOT_DIR, "logs")

# Valid threats to log
LOGGABLE_THREATS = ["violence", "shooting"]
LOG_COOLDOWN = 3.0 # Seconds between logs for the same person

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

# --- 0. SETUP LOGGING ---
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "threat_alerts.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("SmartVision")

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

def process_source(source_path, output_path, detector, lstm_model, device, inv_class_map):
    """
    Handles inference for a single video source (file or webcam).
    """
    is_webcam = source_path == WEBCAM_ID
    source_name = "Webcam" if is_webcam else f"File: {os.path.basename(source_path)}"
    
    print(f"🎬 Starting processing for: {source_name}")
    
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        logger.error(f"Cannot open source {source_name}")
        return

    # Helper Classes per source instance
    tracker = ByteTracker()
    pose_estimator = MediaPipeEstimator()
    person_history = {} 
    
    # Logging Rate Limiter: { track_id: last_log_timestamp }
    last_log_time = {}

    # Video Writer Output (Only if not webcam, or if user wants to record webcam too)
    out_writer = None
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps): fps = 20.0 # Default fallback
        
        out_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        print(f"   recording to: {output_path}")

    while True:
        ret, frame = cap.read()
        if not ret: 
            if not is_webcam: print("   End of stream.")
            break

        h_img, w_img, _ = frame.shape
        results = detector.predict(frame, conf=CONFIDENCE_THRESHOLD)
        
        detections_for_tracking = []
        visible_weapons = [] 

        # --- DETECTION LOOP ---
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = detector.class_names.get(cls_id, "Unknown")

                if cls_id in WEAPON_CLASS_IDS:
                    visible_weapons.append(cls_id)
                    color = get_color(cls_id)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"THREAT: {class_name}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + tw, y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                elif cls_id == PERSON_CLASS_ID:
                    detections_for_tracking.append(([x1, y1, x2, y2], conf, cls_id))

        # --- TRACKING & LSTM LOOP ---
        tracked_objects = tracker.update(detections_for_tracking)

        if len(tracked_objects) > 0:
            for i in range(len(tracked_objects)):
                x1, y1, x2, y2 = map(int, tracked_objects.xyxy[i])
                track_id = int(tracked_objects.tracker_id[i])

                if track_id not in person_history:
                    person_history[track_id] = deque(maxlen=SEQUENCE_LENGTH)

                x1_c, y1_c = max(0, x1), max(0, y1)
                x2_c, y2_c = min(w_img, x2), min(h_img, y2)
                person_crop = frame[y1_c:y2_c, x1_c:x2_c]

                landmarks_list = None
                if person_crop.size > 0:
                    landmarks_list, landmarks_obj = pose_estimator.predict(person_crop)
                    if landmarks_obj:
                        pose_estimator.visualize(person_crop, landmarks_obj)
                        frame[y1_c:y2_c, x1_c:x2_c] = person_crop 
                
                if landmarks_list:
                    person_history[track_id].append(landmarks_list)
                else:
                    if len(person_history[track_id]) > 0:
                        person_history[track_id].append([0.0] * 132)

                # LSTM Inference
                action_label = "Scanning..."
                action_prob = 0.0
                threat_level = "SAFE"
                box_color = (0, 255, 0)

                if len(person_history[track_id]) == SEQUENCE_LENGTH:
                    seq = np.array(person_history[track_id], dtype=np.float32)
                    seq_tensor = torch.tensor(seq).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        output = lstm_model(seq_tensor)
                        probs = torch.softmax(output, dim=1)
                        top_p, top_class = torch.topk(probs, 1)
                        
                        if top_p.item() > 0.6: 
                            action_label = inv_class_map[top_class.item()]
                            action_prob = top_p.item()

                    # Logic Matrix
                    has_gun = any(w in GUN_CLASS_IDS for w in visible_weapons)
                    
                    if action_label == "shooting" and has_gun:
                        threat_level = "CRITICAL: SHOOTER"
                        box_color = (0, 0, 255)
                    elif action_label == "violence" and (3 in visible_weapons):
                        threat_level = "CRITICAL: KNIFE ATTACK"
                        box_color = (0, 0, 255)
                    elif action_label == "violence":
                        threat_level = "HIGH: FIGHTING"
                        box_color = (0, 165, 255)
                    elif action_label == "shooting":
                        threat_level = "WARN: SUSPICIOUS STANCE"
                        box_color = (0, 255, 255)
                    
                    # --- LOGGING LOGIC ---
                    if action_label in LOGGABLE_THREATS and threat_level != "SAFE":
                        current_time = time.time()
                        last_time = last_log_time.get(track_id, 0)
                        
                        if (current_time - last_time) > LOG_COOLDOWN:
                            log_msg = f"THREAT DETECTED | Source: {source_name} | ID: {track_id} | Action: {action_label.upper()} ({action_prob:.2f}) | Level: {threat_level}"
                            if "CRITICAL" in threat_level:
                                logger.critical(log_msg)
                            elif "HIGH" in threat_level:
                                logger.error(log_msg)
                            else:
                                logger.warning(log_msg)
                            
                            last_log_time[track_id] = current_time

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                info_text = f"ID:{track_id} {action_label}"
                if action_prob > 0: info_text += f" ({action_prob:.0%})"
                cv2.putText(frame, info_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                if threat_level != "SAFE":
                    cv2.putText(frame, threat_level, (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

        if out_writer:
            out_writer.write(frame)
            
        cv2.imshow("SmartVision Pro - Threat Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if out_writer: out_writer.release()
    cv2.destroyAllWindows()


def run_pipeline():
    # --- INITIALIZE MODELS ---
    print(" Initializing SmartVision Pipeline...")
    
    detector = YOLODetector()
    if not os.path.exists(YOLO_MODEL_PATH):
        logger.error(f"YOLO model not found at {YOLO_MODEL_PATH}")
        return
    detector.load_model(YOLO_MODEL_PATH)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    
    print(f"   Loading LSTM Brain from {LSTM_MODEL_PATH}...")
    if not os.path.exists(LSTM_MODEL_PATH):
        logger.error(f"LSTM model not found at {LSTM_MODEL_PATH}")
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
    
    class_map = checkpoint['class_map'] 
    inv_class_map = {v: k for k, v in class_map.items()}

    # --- DETERMINE SOURCE ---
    input_videos = []
    if os.path.exists(INPUT_VIDEO_DIR):
        input_videos = [
            f for f in os.listdir(INPUT_VIDEO_DIR) 
            if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov'))
        ]
    
    if input_videos:
        print(f" Found {len(input_videos)} videos in {INPUT_VIDEO_DIR}")
        for video_file in input_videos:
            src_path = os.path.join(INPUT_VIDEO_DIR, video_file)
            out_path = os.path.join(RESULTS_DIR, f"output_{video_file}")
            
            process_source(src_path, out_path, detector, lstm_model, device, inv_class_map)
            
        print(" All videos processed.")
    else:
        print(" No videos found in input folder. Switching to LIVE WEBCAM.")
        # Optional: You can choose to save webcam output or not. 
        # Passing OUTPUT_VIDEO_PATH from prev config if likely desired, else None
        live_out_path = os.path.join(RESULTS_DIR, "webcam_session.mp4")
        process_source(WEBCAM_ID, live_out_path, detector, lstm_model, device, inv_class_map)

if __name__ == "__main__":
    run_pipeline()