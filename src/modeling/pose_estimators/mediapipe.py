import mediapipe as mp
import cv2
import numpy as np


class MediaPipeEstimator:
    """
    Wrapper for MediaPipe Pose.
    Extracts 33 skeletal landmarks from a person's image.
    """

    def __init__(self):
        print("🦴 Initializing MediaPipe Pose...")
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # model_complexity: 0=Fast, 1=Balanced, 2=Accurate
        # min_detection_confidence: Keep high to avoid noise
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def predict(self, person_crop):
        """
        Input: BGR Image (Crop of a person)
        Output: List of 33 landmarks [x, y, z, visibility, ...]
                Returns None if no pose detected.
        """
        # 1. Convert BGR to RGB (MediaPipe requires RGB)
        img_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        # 3. Process Results
        if results.pose_landmarks:
            # We flatten the data for the LSTM
            # 33 landmarks * 4 values (x, y, z, vis) = 132 features per frame
            landmarks = []
            for lm in results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])

            return landmarks, results.pose_landmarks

        return None, None

    def visualize(self, person_crop, landmarks_object):
        """
        Draws the skeleton on the cropped image in-place.
        """
        self.mp_drawing.draw_landmarks(
            person_crop,
            landmarks_object,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
        )
        return person_crop
