# SmartVision Pro - Real-Time Threat Detection

SmartVision Pro is an advanced real-time computer vision system designed to detect potential security threats such as weapons (guns, knives) and violent actions (fighting, shooting) using a multi-stage pipeline.

##  Features

- **Object Detection**: Identifies persons and weapons (handguns, rifles, knives) using **YOLOv11**.
- **Pose Estimation**: Extracts skeletal landmarks using **MediaPipe**.
- **Action Recognition**: Classifies actions (Normal vs. Violence vs. Shooting) using **Bi-LSTM**.
- **Threat Logic Matrix**: Combines detection + action to determine precise threat levels.
- **Flexible Input**: Supports Video Files and Webcam.
- **Dockerized**: specific CPU support for Mac/Windows, GPU optional.

##  Documentation

- **[Installation Guide](docs/installation.md)**: Detailed instructions for Docker (Recommended) and Local setup.
- **[Usage Guide](docs/usage.md)**: How to run the pipeline, interpret results, and understanding the Threat Logic.

##  Quick Start (Docker)

```bash
docker-compose up --build
```
*See [Installation Guide](docs/installation.md) for prerequisites and GPU setup.*

##  Project Structure

```
smart-vision/
├── data/input_videos/  # Drop video files here
├── logs/               # check threat_alerts.log here
├── results/            # processed videos
├── models/             # Place trained models here
└── src/                # Source code
```

##  Disclaimer
This software is for educational and research purposes. Performance depends on model training and environmental conditions.
