import sys
import os
import time
import datetime
import torch
from ultralytics import YOLO
from roboflow import Roboflow


def main():
    # --- 1. SETUP & HARDWARE CHECK ---
    print("=" * 60)
    print(
        f" STARTED TRAINING AT: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print("=" * 60)

    print("\n Checking Hardware...")
    device = "cpu"
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f" SUCCESS: Found {gpu_count} GPU(s).")
        print(f"   Using: {gpu_name}")
        device = 0  # Use GPU 0
    else:
        print(" WARNING: No GPU detected! Training will be painfully slow.")

    # --- 2. DOWNLOAD DATASET ---
    print("\n Downloading Dataset from Roboflow...")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("smartvision-dlaub").project("raw_v2")
    version = project.version(1)
    dataset = version.download("yolov11")

    data_yaml_path = os.path.join(dataset.location, "data.yaml")
    print(f" Dataset ready at: {data_yaml_path}")

    # --- 3. INITIALIZE MODEL ---
    print("\nLoad YOLOv11s...")
    model = YOLO("yolo11s.pt")

    # --- 4. START TRAINING ---
    print("\n Starting Training (Production Config)...")
    print(
        "   Note: Logs are automatically saved to 'runs/detect/production_run_v1/results.csv'"
    )
    print("=" * 60)

    start_time = time.time()

    # Training Loop
    results = model.train(
        data=data_yaml_path,
        epochs=80,
        imgsz=640,
        batch=64,
        device=device,
        # Performance Settings
        cache="ram",
        workers=12,
        # Project Organization
        project="SmartVision_Final",
        name="production_run_v1",
        # --- AUGMENTATIONS ---
        mosaic=0.8,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
    )

    # --- 5. TIME CALCULATION ---
    end_time = time.time()
    duration = end_time - start_time
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)

    print("=" * 60)
    print(f" Training Complete.")
    print(f" Total Time Taken: {hours} hours, {minutes} minutes.")
    print(f" Results saved to: {results.save_dir}")
    print(f" CSV Metrics: {os.path.join(results.save_dir, 'results.csv')}")


if __name__ == "__main__":
    main()
