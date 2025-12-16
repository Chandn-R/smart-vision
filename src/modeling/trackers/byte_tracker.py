import supervision as sv
import numpy as np


class ByteTracker:
    """
    Wrapper for ByteTrack using the Supervision library.
    Excellent for recovering 'lost' tracks in occluded/blurry scenarios.
    """

    def __init__(self):
        print("⚡ Initializing ByteTrack...")
        # track_activation_threshold: Minimum confidence to start a new track
        # lost_track_buffer: How many frames to remember a lost object
        # minimum_matching_threshold: How strict the overlap matching is
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=60,  # Remember lost people for ~2 seconds
            minimum_matching_threshold=0.8,
            frame_rate=30,
        )

    def update(self, detections_list):
        """
        Input:
            detections_list = list of tuples: ([x1, y1, x2, y2], conf, class_id)
        Output:
            Detections object containing track_id
        """
        if len(detections_list) == 0:
            return []

        xyxy = np.array([d[0] for d in detections_list])
        confidence = np.array([d[1] for d in detections_list])
        class_id = np.array([d[2] for d in detections_list])

        detections = sv.Detections(
            xyxy=xyxy, confidence=confidence, class_id=class_id.astype(int)
        )

        tracked_detections = self.tracker.update_with_detections(detections)

        return tracked_detections
