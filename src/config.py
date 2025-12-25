import os
import sys

# --- PROJECT SETUP ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- PATHS ---
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", os.path.join(ROOT_DIR, "models", "yolo_fine_tune_v2.pt"))
LSTM_MODEL_PATH = os.getenv("LSTM_MODEL_PATH", os.path.join(ROOT_DIR, "models", "lstm_action_recognition_pro_v2.pth"))
INPUT_VIDEO_DIR = os.getenv("INPUT_VIDEO_DIR", os.path.join(ROOT_DIR, "data", "input_videos"))
RESULTS_DIR = os.getenv("RESULTS_DIR", os.path.join(ROOT_DIR, "results"))
LOG_DIR = os.getenv("LOG_DIR", os.path.join(ROOT_DIR, "logs"))

# --- PARAMETERS ---
WEBCAM_ID = 0
CONFIDENCE_THRESHOLD = 0.45
SEQUENCE_LENGTH = 30  # Must match training

# --- THREAT LOGIC ---
LOGGABLE_THREATS = ["violence", "shooting"]
LOG_COOLDOWN = 60.0  # Seconds between logs for the same threat type

# --- CLASS DEFINITIONS ---
CLASS_COLORS = {
    0: (0, 0, 0),      # atm: Black/Gray (Ignore?)
    1: (0, 0, 0),      # backpack: Black (Ignore)
    2: (0, 0, 255),    # gun: Red
    3: (0, 0, 0),      # handbag: Black (Ignore)
    4: (0, 0, 255),    # knife: Red
    5: (0, 255, 0),    # person: Green
}
PERSON_CLASS_ID = 5
WEAPON_CLASS_IDS = [2, 4] # Gun, Knife
GUN_CLASS_IDS = [2]
KNIFE_CLASS_ID = 4
ATM_CLASS_ID = 0
BACKPACK_CLASS_ID = 1
HANDBAG_CLASS_ID = 3

def get_color(class_id):
    return CLASS_COLORS.get(class_id, (255, 255, 255))

