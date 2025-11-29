from deep_sort_realtime.deepsort_tracker import DeepSort


class DeepSortTracker:
    """
    Wrapper for DeepSORT to track people across frames.
    """

    def __init__(self, max_age=70):
        print(" Initializing DeepSORT Tracker...")
        # max_age=30: If a person disappears for 30 frames, forget their ID.
        # n_init=3: We need 3 consecutive detections to confirm a real track (reduces flickering).
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=5,  # <--- Lowered from 3 (starts tracking faster)
            nms_max_overlap=0.3,  # Prevents suppressing valid overlapping boxes
            max_cosine_distance=0.4,  # Stricter Re-ID (0.2 - 0.3 is usually good)
            nn_budget=100,
            override_track_class=None,
            embedder="mobilenet",
            half=True, 
            bgr=True, 
            today=None,
        )

    def update(self, detections, frame):
        """
        Input:
            detections = list of [ [x, y, w, h], confidence, class_name ]
            frame = current video frame (used for visual feature extraction)
        Output:
            list of 'Track' objects (with track_id)
        """
        # DeepSORT updates the tracks based on new detections
        # It looks at the image inside the box to re-identify people who crossed paths
        tracks = self.tracker.update_tracks(detections, frame=frame)
        return tracks
