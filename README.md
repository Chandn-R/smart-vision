# SmartVision Pro - Real-Time Threat Detection

SmartVision Pro is an advanced real-time computer vision system designed to detect potential security threats such as weapons (guns, knives) and violent actions (fighting, shooting) using a multi-stage pipeline.

##  Quick Start

### 1. Prerequisites
- **Python 3.11**
- **Virtual Environment**: Expected at `env/` (see "Setup" below).

### 2. Run the Application
We have provided a helper script to start both the **Backend Server** (for alerts) and the **Streamlit UI** (for visualization).

```bash
./start.sh
```

This will launch:
- **Backend Server**: `http://localhost:8000`
- **Dashboard UI**: `http://localhost:8501`

*(Press `Ctrl+C` in the terminal to stop all services)*

---

##  Installation & Setup (First Time)

If you have not set up the project yet, follow these steps:

### 1. Project Setup
```bash
# Clone the repository
git clone <repo_url>
cd smart-vision

# Create Virtual Environment
python3 -m venv env

# Activate Environment
source env/bin/activate

# Install Dependencies
pip install -r environment/requirements.txt
```

### 2. Verify Models
Ensure your trained models are placed in the `models/` directory:
- `models/yolo_fine_tune_v3.pt` (Object Detection)
- `models/lstm_action_recognition_pro_v2.pth` (Action Recognition)

### 3. Start the System
```bash
./start.sh
```

---

##  Project Structure

```
smart-vision/
├── data/
│   ├── input_videos/       # Place video files here for processing
│   ├── incidents.db        # SQLite database for alerts
│   └── processed/          # Intermediate output
├── env/                    # Virtual Environment
├── logs/                   # System and Threat logs
├── models/                 # ML Models (YOLO, LSTM)
├── pipeline/
│   ├── app.py              # Streamlit Dashboard Entry Point
│   └── run_inference.py    # Core ML Inference Logic
├── server/
│   ├── main.py             # FastAPI Backend Entry Point
│   └── config.py           # Configuration
├── start.sh                # One-click startup script
└── results/                # Processed video output
```

##  Usage Guide

### Dashboard (UI)
- **Live Dashboard**: Shows the real-time processed video feed.
- **Alert History**: View past threat alerts saved in the database.
- **Input Source**: Select "Live Webcam" or upload a video file via the sidebar.

### Input Modes
- **Video Files**: Place `.mp4` files in `data/input_videos/`. The pipeline will process them automatically.
- **Webcam**: If no files are selected, the system defaults to the webcam.

##  Notes
- **Performance**: Runs best on machines with GPU support (CUDA/MPS).
- **Security**: The `incidents.db` stores sensitive alert data locally.
