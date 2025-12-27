import os
import sys

# --- PROJECT SETUP ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- PATHS ---
YOLO_MODEL_PATH = os.path.join(ROOT_DIR, "models", "yolo11s_fine_tune_v3.pt")
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
    0: (128, 128, 128),  # ATM: Gray (Non-threat)
    1: (128, 128, 128),  # Backpack: Gray (Non-threat)
    2: (0, 0, 255),    # Gun: Red
    3: (128, 128, 128),  # Handbag: Gray (Non-threat)
    4: (0, 0, 255),    # Knife: Red
    5: (0, 255, 0),    # Person: Green
}
PERSON_CLASS_ID = 5
WEAPON_CLASS_IDS = [2, 4] # Gun, Knife
GUN_CLASS_IDS = [2]  # Specific IDs for firearms
KNIFE_CLASS_ID = 4


def get_color(class_id):
    return CLASS_COLORS.get(class_id, (255, 255, 255))
