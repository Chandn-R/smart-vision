import sys
import os
import cv2
import random

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(ROOT_DIR)
sys.path.append(ROOT_DIR)

from src.modeling.detectors.yolo_detector import YOLODetector
from src.modeling.trackers.byte_tracker import ByteTracker

MODEL_PATH = "../models/yolov11s_fine_tune.pt"
WEBCAM_ID = 0
CONFIDENCE_THRESHOLD = 0.4

CLASS_COLORS = {
    0: (0, 255, 0),  # Person: Green
    1: (0, 0, 255),  # Gun: Red
    2: (0, 0, 255),  # Long Gun: Red
    3: (0, 0, 255),  # Knife: Red
    4: (0, 0, 255),  # Blunt Weapon
    5: (0, 165, 255),  # Burglary Tool: Orange
}

PERSON_CLASS_ID = 0
WEAPON_CLASS_IDS = [1, 2, 3, 4, 5]


def get_color(class_id):
    return CLASS_COLORS.get(class_id, (255, 255, 255))


def run_pipeline():
    detector = YOLODetector()

    if not os.path.exists(MODEL_PATH):
        print(f" Error: Model file not found at {MODEL_PATH}")
        return

    detector.load_model(MODEL_PATH)

    tracker = ByteTracker()

    cap = cv2.VideoCapture(WEBCAM_ID)
    if not cap.isOpened():
        print(f" Error: Cannot open webcam {WEBCAM_ID}")
        return

    print(" Pipeline Started with YOLO Inference and ByteTrack. Press 'Q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.predict(frame, conf=CONFIDENCE_THRESHOLD)

        detections_for_tracking = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = detector.class_names.get(cls_id, "Unknown")

                if cls_id in WEAPON_CLASS_IDS:

                    color = get_color(cls_id)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    label = f" THREAT: {class_name} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                    )
                    cv2.rectangle(frame, (x1, y1 - 30), (x1 + tw, y1), (0, 0, 255), -1)
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )

                    print(f" ALERT: Found {class_name}!")

                elif cls_id == PERSON_CLASS_ID:
                    detections_for_tracking.append(([x1, y1, x2, y2], conf, cls_id))

        tracked_objects = tracker.update(detections_for_tracking)
        if len(tracked_objects) > 0:
            for i in range(len(tracked_objects)):
                x1, y1, x2, y2 = map(int, tracked_objects.xyxy[i])
                track_id = int(tracked_objects.tracker_id[i])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"ID: {track_id} Person"
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        cv2.imshow("SmartVision - ByteTrack", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == "__main__":
    run_pipeline()
