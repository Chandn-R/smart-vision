# SmartVision Pro - Real-Time Threat Detection

SmartVision Pro is an advanced real-time computer vision system designed to detect potential security threats such as weapons (guns, knives) and violent actions (fighting, shooting) using a multi-stage pipeline.

##  Features

- **Object Detection**: Identifies persons and weapons (handguns, rifles, knives, etc.) using a fine-tuned **YOLOv11** model.
- **Pose Estimation**: Extracts skeletal landmarks using **MediaPipe** for detailed movement analysis.
- **Action Recognition**: Uses a bidirectional **LSTM** neural network to classify actions (Normal vs. Violence vs. Shooting) based on 30-frame movement sequences.
- **Threat Logic Matrix**: Combines object detection and action recognition to determine threat levels (e.g., *shooting stance* + *gun detected* = **CRITICAL THREAT**).
- **Flexible Input**: Automatically processes video files from a folder or falls back to live webcam feed.
- **Logging**: Records high-priority threats to `logs/threat_alerts.log` and the console.

##  Installation

1.  **Clone the repository**:
    ```bash
    git clone <repo_url>
    cd smart-vision
    ```

2.  **Install Dependencies**:
    *It is recommended to use a virtual environment.*
    ```bash
    pip install -r requirements.txt
    ```

3.  **Models**:
    Place your trained models in the `models/` directory:
    - `models/yolo_fine_tune_v2.pt`
    - `models/lstm_action_recognition_pro_v2.pth`

##  Usage

### 1. Run the Pipeline
The main inference script is located in `pipeline/run_inference.py`.

```bash
python pipeline/run_inference.py
```

### 2. Input Modes
The system automatically determines the input source:
- **Video Mode**: If you place video files (`.mp4`, `.avi`, `.mkv`) in `data/input_videos/`, the system will process them sequentially and save the output to `results/`.
- **Webcam Mode**: If `data/input_videos/` is empty or does not exist, the system switches to the default webcam (`ID 0`).

### 3. Output
- **Visual**: A window showing the video feed with bounding boxes, pose skeletons, and threat status.
- **Video Files**: Processed videos are saved to `results/`.
- **Logs**: Threat alerts (Violence/Shooting) are logged to `logs/threat_alerts.log`.

##  Project Structure

```
smart-vision/
├── data/
│   ├── input_videos/       # Place input videos here
│   └── processed/          # Intermediate data
├── logs/
│   └── threat_alerts.log   # Threat detection logs (30s cooldown per threat type)
├── models/
│   ├── yolo_fine_tune_v2.pt              # Updated YOLOv11 Model
│   └── lstm_action_recognition_pro_v2.pth # Updated LSTM Action Model
├── pipeline/
│   └── run_inference.py    # Main Inference Pipeline (Refactored)
├── src/
│   ├── config.py           # Central Configuration (Paths, Thresholds, Cooldowns)
│   ├── core/               # Threat Logic & Managers
│   ├── modeling/           # Classifiers (LSTM), Detectors (YOLO), Trackers, Pose
│   └── utils/              # Shared Utilities (Logger)
├── scripts/
│   ├── yolo11_fine_tune_v2.py # V2 Model Training Script
│   └── ...
└── results/                # Output videos (LFS tracked)
```

##  Disclaimer
This software is for educational and research purposes. Performance depends on model training and environmental conditions.
