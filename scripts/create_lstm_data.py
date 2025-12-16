import os
import cv2
import numpy as np
import sys
from ultralytics import YOLO
from tqdm import tqdm

# --- PROJECT SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

try:
    from src.modeling.pose_estimators.mediapipe_estimator import MediaPipeEstimator
except ImportError:
    print("❌ Error: Could not import MediaPipeEstimator.")
    sys.exit(1)

# --- CONFIGURATION ---
VIDEOS_DIR = os.path.expanduser("~/Desktop/dataset/split_clips") 
OUTPUT_DIR = os.path.expanduser("~/Desktop/dataset/lstm_data")
DEBUG_VIDEO_DIR = os.path.expanduser("~/Desktop/dataset/debug_videos") 

SEQUENCE_LENGTH = 30
STEP_SIZE = 10
YOLO_MODEL_SIZE = "yolov11x.pt" 

# THRESHOLDS
ENERGY_THRESHOLD = 0.08  # Min movement to be considered "Action" (Filters statues)
IOU_THRESHOLD = 0.3      # For simple tracking

def calculate_iou(box1, box2):
    x1_a, y1_a, x2_a, y2_a = box1
    x1_b, y1_b, x2_b, y2_b = box2
    x1_i = max(x1_a, x1_b)
    y1_i = max(y1_a, y1_b)
    x2_i = min(x2_a, x2_b)
    y2_i = min(y2_a, y2_b)
    if x2_i < x1_i or y2_i < y1_i: return 0.0
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0

def calculate_pose_energy(landmarks):
    """Simple heuristic: Sum of limb distances from body center."""
    if landmarks is None: return 0.0
    def get_pt(idx): return np.array([landmarks[idx*4], landmarks[idx*4+1]])
    
    wrists = [get_pt(15), get_pt(16)]
    shoulders = [get_pt(11), get_pt(12)]
    hips = [get_pt(23), get_pt(24)]
    
    torso_height = np.linalg.norm(shoulders[0] - hips[0]) + np.linalg.norm(shoulders[1] - hips[1])
    if torso_height == 0: return 0.0
    
    center = (hips[0] + hips[1]) / 2
    energy = 0
    for limb in wrists:
        energy += np.linalg.norm(limb - center) / torso_height
    return energy

def extract_features():
    print(f"⏳ Loading YOLO Model ({YOLO_MODEL_SIZE})...")
    yolo_model = YOLO(YOLO_MODEL_SIZE)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
         yolo_model.to('mps')

    pose_estimator = MediaPipeEstimator()
    
    all_tasks = []
    classes = [d for d in os.listdir(VIDEOS_DIR) if os.path.isdir(os.path.join(VIDEOS_DIR, d))]
    for class_name in classes:
        files = [f for f in os.listdir(os.path.join(VIDEOS_DIR, class_name)) if f.endswith(('.mp4', '.avi', '.mkv'))]
        for f in files:
            all_tasks.append({'path': os.path.join(VIDEOS_DIR, class_name, f), 'class': class_name, 'name': f})
            
    for task in tqdm(all_tasks, desc="Multi-Person Extraction"):
        video_path, class_name, video_name = task['path'], task['class'], task['name']
        cap = cv2.VideoCapture(video_path)
        
        # Debug Video
        width, height = int(cap.get(3)), int(cap.get(4))
        debug_path = os.path.join(DEBUG_VIDEO_DIR, class_name, f"debug_{video_name}")
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        out = cv2.VideoWriter(debug_path, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(5)), (width, height))
        
        # ACTIVE TRACKS: { track_id: [landmark_list] }
        active_tracks = {}
        next_track_id = 0
        
        # PREVIOUS BOXES: { track_id: box } for IoU matching
        prev_boxes = {}
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            results = yolo_model(frame, classes=[0], verbose=False)
            
            current_boxes = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                current_boxes.append((x1, y1, x2, y2))
            
            # --- SIMPLE TRACKING (Match to previous frame) ---
            matched_ids = set()
            new_prev_boxes = {}
            
            for curr_box in current_boxes:
                # Find best match in prev_boxes
                best_iou = 0
                best_id = -1
                
                for tid, prev_box in prev_boxes.items():
                    if tid in matched_ids: continue # Already matched
                    iou = calculate_iou(curr_box, prev_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_id = tid
                
                # Assign ID
                if best_iou > IOU_THRESHOLD:
                    track_id = best_id
                    matched_ids.add(track_id)
                else:
                    track_id = next_track_id
                    next_track_id += 1
                    active_tracks[track_id] = [] # Start new history
                
                new_prev_boxes[track_id] = curr_box
                
                # --- PROCESS PERSON ---
                x1, y1, x2, y2 = curr_box
                h, w, _ = frame.shape
                pad = 10
                cx1, cy1 = max(0, x1-pad), max(0, y1-pad)
                cx2, cy2 = min(w, x2+pad), min(h, y2+pad)
                crop = frame[cy1:cy2, cx1:cx2]
                
                landmarks = None
                energy = 0.0
                if crop.size > 0:
                    landmarks, _ = pose_estimator.predict(crop)
                    if landmarks:
                        energy = calculate_pose_energy(landmarks)
                
                # SAVE DATA (Only if we have landmarks)
                if landmarks:
                    active_tracks[track_id].append({'lm': landmarks, 'energy': energy})
                else:
                    # Pad if missed, but don't increment energy
                    if len(active_tracks[track_id]) > 0:
                         active_tracks[track_id].append({'lm': [0.0]*132, 'energy': 0.0})

                # DEBUG DRAW
                color = (0, 255, 0) if energy > ENERGY_THRESHOLD else (0, 100, 100) # Bright Green = Saved
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{track_id} E:{energy:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            prev_boxes = new_prev_boxes
            out.write(frame)

        cap.release()
        out.release()
        
        # --- SAVE FILTERED TRACKS ---
        for tid, track_data in active_tracks.items():
            # Filter 1: Must be long enough
            if len(track_data) < SEQUENCE_LENGTH: continue
            
            # Filter 2: Must be "Violent" (Avg Energy check) if class is violence
            # If class is Normal, we keep everything.
            avg_energy = np.mean([t['energy'] for t in track_data])
            
            if class_name == 'violence' and avg_energy < ENERGY_THRESHOLD:
                continue # Skip bystander
                
            # Convert to numpy array
            landmarks_only = [t['lm'] for t in track_data]
            data_points = np.array(landmarks_only)
            
            # Sliding Window
            num_sequences = 0
            for i in range(0, len(data_points) - SEQUENCE_LENGTH, STEP_SIZE):
                sequence = data_points[i : i + SEQUENCE_LENGTH]
                save_name = f"{class_name}_{video_name.split('.')[0]}_id{tid}_{num_sequences}.npy"
                np.save(os.path.join(OUTPUT_DIR, class_name, save_name), sequence)
                num_sequences += 1

if __name__ == "__main__":
    import torch
    extract_features()