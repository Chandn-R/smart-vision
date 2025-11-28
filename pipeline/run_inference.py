import sys
import os
import cv2
import random

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(ROOT_DIR)
sys.path.append(ROOT_DIR)

from src.modeling.detectors.yolo_detector import YOLODetector

MODEL_PATH = '../models/yolov11s_fine_tune.pt' 
WEBCAM_ID = 0
CONFIDENCE_THRESHOLD = 0.5

CLASS_COLORS = {
    0: (0, 255, 0),    # Person: Green
    1: (0, 0, 255),    # Gun: Red
    2: (0, 0, 255),    # Long Gun: Red
    3: (0, 0, 255),    # Knife: Red
    5: (0, 165, 255),  # Burglary Tool: Orange
}

def get_color(class_id):
    return CLASS_COLORS.get(class_id, (255, 255, 255))

def run_pipeline():
    detector = YOLODetector()
    
    if not os.path.exists(MODEL_PATH):
        print(f" Error: Model file not found at {MODEL_PATH}")
        return

    detector.load_model(MODEL_PATH)

    cap = cv2.VideoCapture(WEBCAM_ID)
    if not cap.isOpened():
        print(f" Error: Cannot open webcam {WEBCAM_ID}")
        return

    print(" YOLO Inference Started. Press 'Q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        results = detector.predict(frame, conf=CONFIDENCE_THRESHOLD)
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = detector.class_names.get(cls_id, "Unknown")

                color = get_color(cls_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"{class_name} {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        cv2.imshow("SmartVision - YOLO Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

if __name__ == "__main__":
    run_pipeline()