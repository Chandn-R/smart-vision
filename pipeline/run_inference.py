import sys
import os
import cv2
import numpy as np
import torch
import argparse
from collections import deque


# Add root to sys.path to ensure imports work if run from pipeline/ or root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.config import (
    YOLO_MODEL_PATH, LSTM_MODEL_PATH, INPUT_VIDEO_DIR, RESULTS_DIR,
    WEBCAM_ID, CONFIDENCE_THRESHOLD, SEQUENCE_LENGTH,
    PERSON_CLASS_ID, WEAPON_CLASS_IDS, get_color
)
from src.modeling.detectors.yolo_detector import YOLODetector
from src.modeling.trackers.byte_tracker import ByteTracker
from src.modeling.pose_estimators.mediapipe import MediaPipeEstimator
from src.modeling.classifiers.lstm import BidirectionalLSTM
from src.core.threat_manager import ThreatManager
from src.utils.logger import setup_logger

# Initialize global logger
logger = setup_logger()

def process_source(source_path, output_path, detector, lstm_model, device, inv_class_map, threat_manager, headless=False):

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

    # Video Writer Output
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
                    # Removed text labels for weapons (boxes only)
                
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
                
                # --- THREAT LOGIC & LOGGING ---
                threat_level, box_color = threat_manager.determine_threat(action_label, visible_weapons)
                threat_manager.log_threat(track_id, source_name, action_label, action_prob, threat_level)
                
                # Visualization
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                # Removed text annotations as per user request (logs only)
        
        if out_writer:
            out_writer.write(frame)
            
        if not headless:
            cv2.imshow("SmartVision Pro - Threat Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if out_writer: out_writer.release()
    cv2.destroyAllWindows()

def run_pipeline(headless=False):

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
    
    # Initialize ThreatManager
    threat_manager = ThreatManager()

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
            
            process_source(src_path, out_path, detector, lstm_model, device, inv_class_map, threat_manager, headless=headless)
            
        print(" All videos processed.")
    else:
        print(" No videos found in input folder. Switching to LIVE WEBCAM.")
        live_out_path = os.path.join(RESULTS_DIR, "webcam_session.mp4")
        process_source(WEBCAM_ID, live_out_path, detector, lstm_model, device, inv_class_map, threat_manager, headless=headless)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartVision Pro Inference Pipeline")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no GUI)")
    args = parser.parse_args()
    
    run_pipeline(headless=args.headless)