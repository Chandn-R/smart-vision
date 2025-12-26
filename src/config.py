import os
import sys

# --- PROJECT SETUP ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- PATHS ---
YOLO_MODEL_PATH = os.path.join(ROOT_DIR, "models", "yolo_fine_tune_v2.pt")
LSTM_MODEL_PATH = os.path.join(ROOT_DIR, "models", "lstm_action_recognition_pro_v2.pth")
INPUT_VIDEO_DIR = os.path.join(ROOT_DIR, "data", "input_videos")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
LOG_DIR = os.path.join(ROOT_DIR, "logs")

# --- PARAMETERS ---
WEBCAM_ID = 0
CONFIDENCE_THRESHOLD = 0.45
SEQUENCE_LENGTH = 30  # Must match training

# --- THREAT LOGIC ---
LOGGABLE_THREATS = ["violence", "shooting"]
LOG_COOLDOWN = 60.0  # Seconds between logs for the same threat type

# --- CLASS DEFINITIONS ---
CLASS_COLORS = {
    0: (0, 255, 0),    # Person: Green
    1: (0, 0, 255),    # Gun: Red
    2: (0, 0, 255),    # Long Gun: Red
    3: (0, 0, 255),    # Knife: Red
    4: (0, 165, 255),  # Blunt Weapon: Orange
    5: (0, 165, 255),  # Burglary Tool: Orange
}
PERSON_CLASS_ID = 0
WEAPON_CLASS_IDS = [1, 2, 3, 4, 5]
GUN_CLASS_IDS = [1, 2]  # Specific IDs for firearms

def get_color(class_id):
    return CLASS_COLORS.get(class_id, (255, 255, 255))
