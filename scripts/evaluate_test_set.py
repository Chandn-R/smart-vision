import os
import sys
import torch
from roboflow import Roboflow
from ultralytics import YOLO

# Add root directory to path for imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    
from src.config import YOLO_MODEL_PATH

# --- CONFIGURATION ---
# 1. Roboflow Setup (Updated with your specific details)
# API_KEY = ""  #  Ideally, load this from os.getenv("ROBOFLOW_API_KEY") for security
WORKSPACE = "v2-nfgdu"
PROJECT = "the_final"
VERSION = 4 

def main():
    # --- STEP 1: DETECT MAC GPU ---
    device = 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
        print(f" SUCCESS: Apple Silicon GPU detected. Using device='{device}'")
    elif torch.cuda.is_available():
        device = 0
        print(f" SUCCESS: NVIDIA GPU detected. Using device='{device}'")
    else:
        print(" WARNING: No GPU detected. Running on CPU (this will be slow).")

    # --- STEP 2: DOWNLOAD DATASET ---
    print("\n⬇  Downloading Dataset from Roboflow...")
    
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    version = project.version(VERSION)
    
    # Download ONCE
    dataset = version.download("yolov11")
    
    print(f"\n Dataset downloaded to: {dataset.location}")
    
    # --- STEP 3: SETUP PATHS ---
    data_yaml_path = os.path.join(dataset.location, "data.yaml")
    
    if not os.path.exists(data_yaml_path):
        print(f" Critical Error: data.yaml not found at {data_yaml_path}")
        return

    # --- STEP 4: RUN TEST EVALUATION ---
    print("\n Starting YOLOv11 Test Evaluation...")
    print(f"   Target Model: {YOLO_MODEL_PATH}")
    print(f"   Target Data:  {data_yaml_path}")
    print("=" * 60)

    # Check if model exists
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f" Error: Model weights not found at {YOLO_MODEL_PATH}")
        return

    model = YOLO(YOLO_MODEL_PATH)

    # Run Validation on the TEST split
    metrics = model.val(
        data=data_yaml_path,
        split='test',         # <--- Force YOLO to use the Test set
        imgsz=640,
        batch=16,
        conf=0.001,           # Low confidence for accurate mAP calculation
        iou=0.6,
        
        # --- CRITICAL FOR MAC ---
        device=device,
        # ------------------------
        
        project="SmartVision_Final",
        name="test_evaluation_final",
        plots=True
    )

    # --- STEP 5: RESULTS ---
    print("\n" + "=" * 60)
    print("🎯 FINAL TEST SPLIT RESULTS")
    print("=" * 60)
    print(f"  • mAP @ 0.5:    {metrics.box.map50:.4f}")
    print(f"  • mAP @ 0.5:95: {metrics.box.map:.4f}")
    print(f"  • Precision:    {metrics.box.mp:.4f}")
    print(f"  • Recall:       {metrics.box.mr:.4f}")
    print("-" * 60)
    print(f"📁 Plots saved to: {metrics.save_dir}")

if __name__ == "__main__":
    main()