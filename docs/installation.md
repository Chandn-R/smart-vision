# Installation Guide

SmartVision Pro can be deployed via **Docker** (Recommended for Production) or installed locally (Development).

## 🐳 Docker Deployment (Recommended)

This method ensures cross-platform compatibility (Linux, Mac, Windows) and handles all system dependencies automatically.

### Prerequisites
- Docker Desktop or Docker Engine installed.
- Trained models placed in the `models/` directory:
    - `models/yolo_fine_tune_v2.pt`
    - `models/lstm_action_recognition_pro_v2.pth`

### Quick Start
1. **Build and Run**:
   ```bash
   docker-compose up --build
   ```
   *Note: This defaults to CPU mode. For GPU support, see [GPU Configuration](#gpu-configuration).*

2. **Watch Logs**:
   ```bash
   docker-compose logs -f
   ```

3. **Stop**:
   ```bash
   docker-compose down
   ```

### GPU Configuration
By default, the `docker-compose.yml` is configured for specific CPU execution to ensure maximum compatibility. 
To enable **NVIDIA GPU** support:
1. Open `docker-compose.yml`.
2. Uncomment the `deploy` section under `resources`.
3. Restart the container.

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
