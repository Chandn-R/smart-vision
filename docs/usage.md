# Usage Guide

## Running the Pipeline

### Docker Mode
The pipeline runs automatically when the container starts. It uses the arguments defined in `Dockerfile` (default: `--headless`).

### Local Mode
Run the inference script directly:
```bash
python pipeline/run_inference.py [OPTIONS]
```

**Options**:
- `--headless`: Run without GUI (for servers/Docker).

---

## Input Modes
The system automatically determines the input source:

1.  **Video Mode**: 
    - Place video files (`.mp4`, `.avi`, `.mkv`) in `data/input_videos/`.
    - The system processes them sequentially.
    - Output saved to: `results/`.

2.  **Webcam Mode**:
    - If `data/input_videos/` is empty, the system switches to the default webcam (`ID 0`).
    - *Note: Webcam mode requires GUI access (local execution) or device mapping in Docker.*

---

## Output & Validation

### 1. Visual Output (`results/`)
Processed videos will contain:
- **Bounding Boxes**:
  - 🟢 **Green**: Safe / Person
  - 🔴 **Red**: Critical Threat (Gun, Knife, Shooter)
  - 🟠 **Orange**: High Warning (Fighting)
  - 🟡 **Yellow**: Warning (Suspicious Stance, Weapon presence)

- **Annotations**:
  - Above head: `Action: <Type> (<Probability>%)`
  - Below feet: `THREAT: <Level>`

### 2. Threat Logs (`logs/threat_alerts.log`)
Crucial for headless deployments.
```text
2024-12-25 14:30:05 - CRITICAL - THREAT DETECTED | Source: File: fight.mp4 | ID: 12 | Action: VIOLENCE (0.95) | Level: CRITICAL: KNIFE ATTACK
```

### 3. Hierarchical Threat Logic Matrix
The system uses a strict rule set to determine threat levels:

| Priority | Threat Level | Condition A (Object) | Condition B (Action) |
| :--- | :--- | :--- | :--- |
| 1 | **CRITICAL** | Gun Detected | "Shooting" Action |
| 2 | **CRITICAL** | Knife Detected | "Violence" Action |
| 3 | **HIGH** | No Weapon | "Violence" Action |
| 4 | **WARNING** | No Weapon | "Shooting" Stance |
| 5 | **WARNING** | Weapon Detected | Normal / Passive Behavior |
| 6 | **SAFE** | No Weapon | Normal Behavior |
