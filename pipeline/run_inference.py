# run_inference.py
import os
import sys
import time
from collections import deque
from typing import Dict, Deque, List, Tuple, Optional, Union

import cv2
import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.config import (
    YOLO_MODEL_PATH,
    LSTM_MODEL_PATH,
    INPUT_VIDEO_DIR,
    RESULTS_DIR,
    WEBCAM_ID,
    CONFIDENCE_THRESHOLD,
    ACTION_CONFIDENCE_THRESHOLD,
    SEQUENCE_LENGTH,

    PERSON_CLASS_ID,
    WEAPON_CLASS_IDS,
    CONTEXT_CLASS_IDS,
    UNATTENDED_BAG_THRESHOLD,
    get_color,
    DRAW_PERSON_BOXES,
    DRAW_LABELS,
)


from src.modeling.detectors.yolo_detector import YOLODetector
from src.modeling.trackers.byte_tracker import ByteTracker
from src.modeling.pose_estimators.mediapipe import MediaPipeEstimator
from src.modeling.classifiers.lstm import BidirectionalLSTM
from src.core.threat_manager import ThreatManager
from src.utils.logger import setup_logger

logger = setup_logger()


# ---------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------
def load_models():
    """
    Load YOLO detector and LSTM action classifier, return:
    detector, lstm_model, device, inv_class_map, threat_manager
    """
    # YOLO
    detector = YOLODetector()
    if not os.path.exists(YOLO_MODEL_PATH):
        logger.error(f"YOLO model not found at {YOLO_MODEL_PATH}")
        return None, None, None, None, None
    detector.load_model(YOLO_MODEL_PATH)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")

    # LSTM
    if not os.path.exists(LSTM_MODEL_PATH):
        logger.error(f"LSTM model not found at {LSTM_MODEL_PATH}")
        return None, None, None, None, None

    checkpoint = torch.load(LSTM_MODEL_PATH, map_location=device)
    lstm_model = BidirectionalLSTM(
        checkpoint["input_size"],
        checkpoint["hidden_size"],
        checkpoint["num_layers"],
        len(checkpoint["class_map"]),
    ).to(device)
    lstm_model.load_state_dict(checkpoint["model_state_dict"])
    lstm_model.eval()

    class_map = checkpoint["class_map"]
    inv_class_map = {v: k for k, v in class_map.items()}

    # Threat manager (shared)
    threat_manager = ThreatManager()

    return detector, lstm_model, device, inv_class_map, threat_manager


# ---------------------------------------------------------------------
# LOW-LEVEL HELPERS (DETECTION, TRACKING, LSTM, THREAT)
# ---------------------------------------------------------------------
def run_yolo_detection(detector: YOLODetector, frame: np.ndarray):
    """Run YOLO and return raw results."""
    return detector.predict(frame, conf=CONFIDENCE_THRESHOLD)


def parse_detections(detector: YOLODetector, results) -> Tuple[List, List[int]]:
    """
    From YOLO results, prepare:
    - detections_for_tracking: list of ([x1, y1, x2, y2], conf, cls_id) for persons
    - visible_weapons: list of weapon class ids
    """
    detections_for_tracking: List[Tuple[List[int], float, int]] = []
    visible_weapons: List[int] = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            if cls_id in WEAPON_CLASS_IDS:
                visible_weapons.append(cls_id)
            elif cls_id == PERSON_CLASS_ID:
                detections_for_tracking.append(([x1, y1, x2, y2], conf, cls_id))

    return detections_for_tracking, visible_weapons


def draw_weapon_boxes(detector: YOLODetector, frame: np.ndarray, results) -> List[Dict]:
    """
    Draw bounding boxes for weapons and context objects (ATM, Bags) and return detailed detection list.
    """
    visible_objects: List[Dict] = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if (cls_id in WEAPON_CLASS_IDS) or (cls_id in CONTEXT_CLASS_IDS):
                class_name = detector.class_names.get(cls_id, "Unknown")
                
                # Append detailed info
                visible_objects.append({
                    "class": class_name,
                    "class_id": cls_id,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })
                
                color = get_color(cls_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                if DRAW_LABELS:
                    label = f"{class_name}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + tw, y1), color, -1)
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

    return visible_objects


def update_person_history(
    pose_estimator: MediaPipeEstimator,
    frame: np.ndarray,
    tracked_objects,
    person_history: Dict[int, Deque[List[float]]],
    device,
    lstm_model,
    inv_class_map,
    threat_manager: ThreatManager,
    context_detections: List[Dict],
    person_durations: Dict[int, float],
    is_unattended_bag: bool,
    frame_id: int = 0
):
    """
    Core per-frame logic: pose estimation, LSTM action recognition,
    threat determination, and drawing person boxes.

    Returns a list of threat messages for UI logging.
    """
    h_img, w_img, _ = frame.shape
    threat_messages: List[str] = []

    if len(tracked_objects) == 0:
        return threat_messages

    # Extract IDs for logic
    visible_object_ids = [d['class_id'] for d in context_detections]

    from src.config import ATM_CLASS_ID
    has_atm = ATM_CLASS_ID in visible_object_ids

    for i in range(len(tracked_objects)):
        x1, y1, x2, y2 = map(int, tracked_objects.xyxy[i])
        track_id = int(tracked_objects.tracker_id[i])
        # Try to get confidence if available, else default
        conf = float(tracked_objects.confidence[i]) if hasattr(tracked_objects, 'confidence') and tracked_objects.confidence is not None else 0.0

        if track_id not in person_history:
            person_history[track_id] = deque(maxlen=SEQUENCE_LENGTH)

        # Crop person
        x1_c, y1_c = max(0, x1), max(0, y1)
        x2_c, y2_c = min(w_img, x2), min(h_img, y2)
        person_crop = frame[y1_c:y2_c, x1_c:x2_c]

        landmarks_list = None
        if person_crop.size > 0:
            landmarks_list, landmarks_obj = pose_estimator.predict(person_crop)
            if landmarks_obj:
                # pose_estimator.visualize(person_crop, landmarks_obj)
                frame[y1_c:y2_c, x1_c:x2_c] = person_crop

        if landmarks_list:
            person_history[track_id].append(landmarks_list)
        else:
            if len(person_history[track_id]) > 0:
                person_history[track_id].append([0.0] * 132)

        # LSTM action recognition
        action_label = "normal"
        action_prob = 0.0

        if len(person_history[track_id]) == SEQUENCE_LENGTH:
            seq = np.array(person_history[track_id], dtype=np.float32)
            seq_tensor = torch.tensor(seq).unsqueeze(0).to(device)

            with torch.no_grad():
                output = lstm_model(seq_tensor)
                probs = torch.softmax(output, dim=1)
                top_p, top_class = torch.topk(probs, 1)

                if top_p.item() > ACTION_CONFIDENCE_THRESHOLD:

                    action_label = inv_class_map[top_class.item()]
                    action_prob = top_p.item()

        # Threat logic
        duration = person_durations.get(track_id, 0.0)
        threat_level, box_color = threat_manager.determine_threat(
            action_label, visible_object_ids, duration, is_unattended_bag
        )
        
        # Prepare Logging Payload
        person_detection = {
            "class": "Person", 
            "confidence": conf,
            "bbox": [x1, y1, x2, y2],
            "track_id": track_id
        }
        all_detections = [person_detection] + context_detections
        spatial_context = {
            "atm_present": has_atm,
            "near_atm": has_atm # Simplified: if in frame, assumed 'near' for now
        }

        threat_manager.log_threat(
            track_id, "Stream", action_label, action_prob, threat_level,
            frame_id=frame_id,
            detections=all_detections,
            spatial_context=spatial_context,
            frame=frame
        )

        # Draw person box and labels (if enabled)
        if DRAW_PERSON_BOXES:
            person_color = (0, 255, 0) # Green
            cv2.rectangle(frame, (x1, y1), (x2, y2), person_color, 2)
            
            if DRAW_LABELS:
                info_text = f"ID:{track_id} ({int(duration)}s)"
                (tw, th), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + tw, y1), person_color, -1)
                cv2.putText(
                    frame,
                    info_text,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

        # Add visual threat label if needed
        if threat_level != "SAFE":
            msg = f" ID {track_id}: {threat_level} ({action_label})"
            threat_messages.append(msg)
            
            if DRAW_LABELS:
                cv2.putText(
                    frame,
                    threat_level,
                    (x1, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    box_color,
                    2,
                )

    return threat_messages


# ---------------------------------------------------------------------
# HIGH-LEVEL INFERENCE LOOPS
# ---------------------------------------------------------------------
def process_frame(
    frame: np.ndarray,
    detector,
    lstm_model,
    device,
    inv_class_map,
    threat_manager: ThreatManager,
    tracker: ByteTracker,
    pose_estimator: MediaPipeEstimator,
    person_history: Dict[int, Deque[List[float]]],
    state_data: Dict = None,
    dt: float = 0.05,
):
    """
    Single-frame processing used by both CLI pipeline and Streamlit UI.

    Returns:
    - processed_frame (BGR)
    - threat_messages (list of strings)
    """
    if state_data is None:
        state_data = {"person_durations": {}, "unattended_bag_timer": 0.0}

    # 1. YOLO
    results = run_yolo_detection(detector, frame)

    # 2. Draw weapons/context and collect visible objects
    context_detections = draw_weapon_boxes(detector, frame, results) # Returns List[Dict]

    # Extract class IDs for internal logic
    visible_object_ids = [d['class_id'] for d in context_detections]

    # Update Unattended Bag Timer
    # Check if there are Bags and NO persons
    from src.config import BACKPACK_CLASS_ID, HANDBAG_CLASS_ID
    has_bag = any(c in [BACKPACK_CLASS_ID, HANDBAG_CLASS_ID] for c in visible_object_ids)
    
    # 3. Prepare detections for tracking
    detections_for_tracking, _ = parse_detections(detector, results)

    # 4. Tracking
    tracked_objects = tracker.update(detections_for_tracking)
    
    num_persons = len(tracked_objects)
    
    # Update state_data based on detections
    if has_bag and num_persons == 0:
        state_data["unattended_bag_timer"] += dt
    else:
        state_data["unattended_bag_timer"] = 0.0
    
    is_unattended_bag = state_data["unattended_bag_timer"] > UNATTENDED_BAG_THRESHOLD

    # Update person durations
    active_ids = []
    for i in range(len(tracked_objects)):
        tid = int(tracked_objects.tracker_id[i])
        active_ids.append(tid)
        state_data["person_durations"][tid] = state_data["person_durations"].get(tid, 0.0) + dt
    
    # Optional: cleanup stats for missing IDs? (Not strictly necessary for short clips, but good practice)

    # 5. Pose + LSTM + Threat
    threat_messages = update_person_history(
        pose_estimator=pose_estimator,
        frame=frame,
        tracked_objects=tracked_objects,
        person_history=person_history,
        device=device,
        lstm_model=lstm_model,
        inv_class_map=inv_class_map,
        threat_manager=threat_manager,
        context_detections=context_detections,
        person_durations=state_data["person_durations"],
        is_unattended_bag=is_unattended_bag,
        frame_id=state_data.get("frame_count", 0) # We need to track frame count
    )

    state_data["frame_count"] = state_data.get("frame_count", 0) + 1

    # Handle separate case: Unattended bag with NO people (threat_messages empty so far)
    if is_unattended_bag and num_persons == 0:
        # We need to manually invoke threat manager or just append message
        # Invoke threat manager to respect logging rules
        # Pass context detections for payload
        spatial_context = { "atm_present": any(d['class_id'] == 0 for d in context_detections), "near_atm": False }
        t_level, t_color = threat_manager.determine_threat("normal", visible_object_ids, 0.0, True)
        if t_level != "SAFE":
             msg = f" {t_level}"
             threat_messages.append(msg)
             threat_manager.log_threat(
                 0, "Stream", "unattended", 1.0, t_level, 
                 frame_id=state_data.get("frame_count", 0),
                 detections=context_detections,
                 spatial_context=spatial_context,
                 frame=frame
            )
             if DRAW_LABELS:
                 # Draw big warning?
                 cv2.putText(frame, t_level, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, t_color, 2)

    return frame, threat_messages


def run_offline_pipeline():
    """
    Original CLI batch pipeline: loop over INPUT_VIDEO_DIR or webcam.
    This is unchanged, but now uses process_frame().
    """
    print(" Initializing SmartVision Pipeline...")

    detector, lstm_model, device, inv_class_map, threat_manager = load_models()
    if detector is None:
        return

    tracker = ByteTracker()
    pose_estimator = MediaPipeEstimator()
    person_history: Dict[int, Deque[List[float]]] = {}

    input_videos = []
    if os.path.exists(INPUT_VIDEO_DIR):
        input_videos = [
            f
            for f in os.listdir(INPUT_VIDEO_DIR)
            if f.lower().endswith((".mp4", ".avi", ".mkv", ".mov"))
        ]

    if input_videos:
        print(f" Found {len(input_videos)} videos in {INPUT_VIDEO_DIR}")
        for video_file in input_videos:
            src_path = os.path.join(INPUT_VIDEO_DIR, video_file)
            out_path = os.path.join(RESULTS_DIR, f"output_{video_file}")
            _run_single_source_offline(
                src_path,
                out_path,
                detector,
                lstm_model,
                device,
                inv_class_map,
                threat_manager,
                tracker,
                pose_estimator,
                person_history,
            )
        print(" All videos processed.")
    else:
        print(" No videos found in input folder. Switching to LIVE WEBCAM.")
        live_out_path = os.path.join(RESULTS_DIR, "webcam_session.mp4")
        _run_single_source_offline(
            WEBCAM_ID,
            live_out_path,
            detector,
            lstm_model,
            device,
            inv_class_map,
            threat_manager,
            tracker,
            pose_estimator,
            person_history,
        )


def _run_single_source_offline(
    source_path,
    output_path,
    detector,
    lstm_model,
    device,
    inv_class_map,
    threat_manager,
    tracker,
    pose_estimator,
    person_history,
):
    is_webcam = source_path == WEBCAM_ID
    source_name = "Webcam" if is_webcam else f"File: {os.path.basename(source_path)}"

    print(f"🎬 Starting processing for: {source_name}")

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        logger.error(f"Cannot open source {source_name}")
        return

    out_writer = None
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps):
            fps = 20.0

        out_writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        print(f"   recording to: {output_path}")

        print(f"   recording to: {output_path}")

    state_data = {
        "person_durations": {},
        "unattended_bag_timer": 0.0
    }
    last_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if not is_webcam:
                print("   End of stream.")
            break

        # Time delta
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        
        # If webcam, dt is real time. If file, try to limit or use fixed?
        # For simplicity in offline file, we can use 1/FPS if we wanted to simulate real speed,
        # but here we just process as fast as possible. 
        # However, to simulate duration properly, we should use 1/FPS for video files 
        # and real time for webcam.
        if is_webcam:
            pass # dt is correct
        elif fps > 0:
            dt = 1.0 / fps # fixed step for files
        
        frame, _ = process_frame(
            frame,
            detector,
            lstm_model,
            device,
            inv_class_map,
            threat_manager,
            tracker,
            pose_estimator,
            person_history,
            state_data=state_data,
            dt=dt
        )

        if out_writer:
            out_writer.write(frame)

        cv2.imshow("SmartVision Pro - Threat Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if out_writer:
        out_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_offline_pipeline()
