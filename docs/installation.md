# Installation Guide

SmartVision Pro can be deployed via **Docker** (Recommended for Production) or installed locally (Development).

## 🐳 Docker Deployment (Recommended)

This method ensures cross-platform compatibility (Linux, Mac, Windows) and handles all system dependencies automatically.

### Prerequisites
- Docker Desktop or Docker Engine installed.
- Trained models placed in the `models/` directory:
    - `models/yolo_fine_tune_v2.pt`
    - `models/lstm_action_recognition_pro_v2.pth`

### Quick Start (CPU / Default)
Works on all platforms (Mac, Windows, Linux).
```bash
docker-compose up --build
```

### Quick Start (Enabled GPU) - Windows/Linux Only
For machines with NVIDIA GPUs:
```bash
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```
*Note: Docker on macOS cannot access the GPU directly. To use the GPU (MPS) on Mac, please use the **Local Installation** method.*

### GPU Configuration Details
- **Windows/Linux**: Use the `docker-compose.gpu.yml` override to pass the NVIDIA GPU to the container.
- **Mac (Apple Silicon)**: Docker runs in a Linux VM that cannot see the Apple Neural Engine. For GPU acceleration, run the pipeline **locally** (`python pipeline/run_inference.py`).

---

## 🐍 Local Installation (Development)

Use this method if you need to modify code or train models.

### Prerequisites
- Python 3.10+
- Virtual Environment (Recommended)

### Steps

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Chandn-R/smart-vision.git
    cd smart-vision
    ```

2. **Create & Activate Virtual Environment**:
    ```bash
    python3 -m venv env
    source env/bin/activate  # Mac/Linux
    # env\Scripts\activate   # Windows
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This installs `opencv-python-headless`. If you need the GUI locally, you may need to install standard `opencv-python`.*

4. **Verify Installation**:
    ```bash
    python pipeline/run_inference.py --help
    ```
