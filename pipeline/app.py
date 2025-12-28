# app.py
import os
import sys
import time
import tempfile
import base64

import cv2
import numpy as np
import torch
import streamlit as st

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.config import (
    INPUT_VIDEO_DIR,
    WEBCAM_ID,
)
from run_inference import (
    load_models,
    process_frame,
    ByteTracker,
    MediaPipeEstimator,
)

from collections import deque


st.set_page_config(
    page_title="SmartVision Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def init_models_and_helpers():
    """
    Wrapper for cached model + helper creation for Streamlit.
    """
    detector, lstm_model, device, inv_class_map, threat_manager = load_models()
    if detector is None:
        return None

    tracker = ByteTracker()
    pose_estimator = MediaPipeEstimator()
    person_history = {}

    return {
        "detector": detector,
        "lstm_model": lstm_model,
        "device": device,
        "inv_class_map": inv_class_map,
        "threat_manager": threat_manager,
        "tracker": tracker,
        "pose_estimator": pose_estimator,
        "person_history": person_history,
    }


def run_streamlit_inference(source_path: str):
    """
    Streamlit-specific loop: handles UI (columns, stats, buttons),
    delegates all heavy logic to run_inference.process_frame().
    """
    runtime = init_models_and_helpers()
    if runtime is None:
        st.error(" Failed to load models. Check paths and restart.")
        return

    detector = runtime["detector"]
    lstm_model = runtime["lstm_model"]
    device = runtime["device"]
    inv_class_map = runtime["inv_class_map"]
    threat_manager = runtime["threat_manager"]
    tracker = runtime["tracker"]
    pose_estimator = runtime["pose_estimator"]
    person_history = runtime["person_history"]

    col1, col2 = st.columns([3, 1])

    with col1:
        st_frame = st.empty()

    with col2:
        st.markdown("### Live Statistics")
        kpi1, kpi2 = st.columns(2)
        with kpi1:
            st.markdown("**Status**")
            status_text = st.markdown("Initializing...")
        with kpi2:
            st.markdown("**FPS**")
            fps_text = st.markdown("0")

        st.markdown("---")
        st.markdown("**Threat Log:**")
        threat_log_container = st.empty()

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        st.error(f"Error opening video source: {source_path}")
        return

    stop_button = st.sidebar.button(" Stop Processing")
    detection_log = []

    prev_time = time.time()

    while cap.isOpened():
        if stop_button:
            st.warning("Processing stopped by user.")
            break

        ret, frame = cap.read()
        if not ret:
            st.success("Video finished.")
            break

        # FPS
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time
        fps_text.write(f"{int(fps)}")
        status_text.write(" Running")

        # Delegate heavy work
        processed_frame, threat_messages = process_frame(
            frame,
            detector,
            lstm_model,
            device,
            inv_class_map,
            threat_manager,
            tracker,
            pose_estimator,
            person_history,
        )

        # Update Threat Log in UI (keep last 5)
        for msg in threat_messages:
            if msg not in detection_log:
                detection_log.insert(0, msg)
        if detection_log:
            threat_log_container.text("\n".join(detection_log[:5]))

        # Show frame (Base64 encoded to prevent MediaFileStorageError)
        _, buffer = cv2.imencode('.jpg', processed_frame)
        monitor_b64 = base64.b64encode(buffer).decode()
        st_frame.markdown(
            f'<img src="data:image/jpeg;base64,{monitor_b64}" style="width:100%">',
            unsafe_allow_html=True
        )

    cap.release()


def main():
    st.sidebar.title("Configuration")

    runtime = init_models_and_helpers()
    if runtime is None or runtime["detector"] is None:
        st.error(" YOLO/LSTM models not found. Check configuration.")
        return

    st.sidebar.success("Models Loaded!")

    input_option = st.sidebar.selectbox(
        "Select Input Source",
        ("Live Webcam", "Upload Video", "Select from Input Folder"),
    )

    source_path = None

    if input_option == "Live Webcam":
        source_path = WEBCAM_ID
        st.info("Using Live Webcam. Click 'Start' in sidebar.")

    elif input_option == "Select from Input Folder":
        if os.path.exists(INPUT_VIDEO_DIR):
            video_files = [
                f
                for f in os.listdir(INPUT_VIDEO_DIR)
                if f.lower().endswith((".mp4", ".avi", ".mov"))
            ]
            if video_files:
                selected_file = st.sidebar.selectbox("Choose a video:", video_files)
                source_path = os.path.join(INPUT_VIDEO_DIR, selected_file)
            else:
                st.sidebar.warning("No videos found in input folder.")
        else:
            st.sidebar.error(f"Input folder {INPUT_VIDEO_DIR} does not exist.")

    elif input_option == "Upload Video":
        uploaded_file = st.sidebar.file_uploader(
            "Upload a video file", type=["mp4", "avi", "mov"]
        )
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            source_path = tfile.name

    if st.sidebar.button("▶️ Start Inference", type="primary"):
        if source_path is not None:
            run_streamlit_inference(source_path)
        else:
            st.error("Please select a valid source.")


if __name__ == "__main__":
    main()
