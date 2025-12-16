# pip install tqdm 
# pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

import os
import cv2
import numpy as np
import sys
from ultralytics import YOLO
from tqdm import tqdm  # <--- NEW: Import the progress bar library

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

def extract_features():
    # 1. Setup Models
    print(f"⏳ Loading YOLO Model ({YOLO_MODEL_SIZE}) on MPS (Mac)...")
    yolo_model = YOLO(YOLO_MODEL_SIZE)
    yolo_model.to('mps')
    
    # Auto-detect Mac GPU (MPS)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
         yolo_model.to('mps')
         print("   ✅ Using Apple M-Series GPU (MPS)")
    else:
         print("   ⚠️ MPS not detected. Using CPU (Will be slower).")

    pose_estimator = MediaPipeEstimator()
    
    # 2. Scan Files First (For the Progress Bar)
    print("🔍 Scanning dataset...")
    if not os.path.exists(VIDEOS_DIR):
        print(f"❌ Error: Input directory '{VIDEOS_DIR}' does not exist.")
        return

    all_tasks = []
    classes = [d for d in os.listdir(VIDEOS_DIR) if os.path.isdir(os.path.join(VIDEOS_DIR, d))]
    
    for class_name in classes:
        class_input_dir = os.path.join(VIDEOS_DIR, class_name)
        # Create output dirs now
        os.makedirs(os.path.join(OUTPUT_DIR, class_name), exist_ok=True)
        os.makedirs(os.path.join(DEBUG_VIDEO_DIR, class_name), exist_ok=True)
        
        files = [f for f in os.listdir(class_input_dir) if f.endswith(('.mp4', '.avi', '.mkv'))]
        for f in files:
            all_tasks.append({
                'path': os.path.join(class_input_dir, f),
                'class': class_name,
                'name': f
            })
            
    print(f"🚀 Found {len(all_tasks)} total videos to process.")
    print("   Starting extraction... (Check the bar below for ETA)")

    # 3. Process with Progress Bar
    # 'tqdm' wraps the loop and handles the timer/ETA automatically
    for task in tqdm(all_tasks, desc="Total Progress", unit="vid"):
        
        video_path = task['path']
        class_name = task['class']
        video_name = task['name']
        
        cap = cv2.VideoCapture(video_path)
        
        # Setup Video Writer
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        debug_path = os.path.join(DEBUG_VIDEO_DIR, class_name, f"debug_{video_name}")
        out_writer = cv2.VideoWriter(debug_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        video_landmarks = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # A. YOLO
            results = yolo_model(frame, classes=[0], verbose=False) 
            
            person_crop = None
            best_box = None
            max_area = 0
            
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    best_box = (x1, y1, x2, y2)

            # B. MediaPipe & Debug Drawing
            landmarks = None
            if best_box:
                x1, y1, x2, y2 = best_box
                
                # Debug: Draw Blue Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                h_img, w_img, _ = frame.shape
                pad = 20
                cx1, cy1 = max(0, x1-pad), max(0, y1-pad)
                cx2, cy2 = min(w_img, x2+pad), min(h_img, y2+pad)
                person_crop = frame[cy1:cy2, cx1:cx2]
                
                if person_crop.size > 0:
                    landmarks, _ = pose_estimator.predict(person_crop)
                    
                    if landmarks:
                         cv2.putText(frame, "Skeleton OK", (x1, y1-10), 
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
            out_writer.write(frame)

            if landmarks:
                video_landmarks.append(landmarks)
            else:
                video_landmarks.append([0.0] * 132) 
        
        cap.release()
        out_writer.release()
        
        # C. Save Data
        data_points = np.array(video_landmarks)
        if len(data_points) >= SEQUENCE_LENGTH:
            num_sequences = 0
            for i in range(0, len(data_points) - SEQUENCE_LENGTH, STEP_SIZE):
                sequence = data_points[i : i + SEQUENCE_LENGTH]
                save_name = f"{class_name}_{video_name.split('.')[0]}_{num_sequences}.npy"
                np.save(os.path.join(OUTPUT_DIR, class_name, save_name), sequence)
                num_sequences += 1

    print("\n✅ Processing Complete! Data saved to 'dataset/lstm_data'")

if __name__ == "__main__":
    import torch # Imported here to check for MPS
    extract_features()