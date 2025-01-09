import streamlit as st
import time
from src.object_detection import detect_objects
import tempfile
import threading

# Shared configuration dictionary for real-time updates
shared_config = {"frame_skip": 1, "confidence_threshold": 0.5}

def update_config():
    # Streamlit sliders for real-time updates
    st.sidebar.title("Settings")
    shared_config["frame_skip"] = st.sidebar.slider("Frame Skip", 1, 10, 1)
    shared_config["confidence_threshold"] = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

def object_detection_stream(video_path):
    # Real-time object detection generator
    frame_skip = shared_config["frame_skip"]
    confidence_threshold = shared_config["confidence_threshold"]

    for result in detect_objects(video_path, frame_skip=frame_skip, confidence_threshold=confidence_threshold):
        yield result

def main():
    st.title("Real-Time Object Detection")

    # Video file uploader
    video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    if video_file is not None:
        # Save uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
            tmpfile.write(video_file.read())
            tmpfile_path = tmpfile.name

        stframe = st.empty()
        object_placeholder = st.empty()

        # Update the shared configuration
        update_config()

        # Process video frames
        detections = object_detection_stream(tmpfile_path)
        for result in detections:
            frame = result["frame"]
            frame_detections = result["detections"]

            # Display the frame
            stframe.image(frame, channels="BGR", use_container_width=True)

            # Display detected objects
            detected_objects = [
                f"**{detection['object']}**: {detection['confidence']:.2f}%"
                for detection in frame_detections
                if detection['confidence'] >= shared_config["confidence_threshold"]
            ]
            object_placeholder.markdown("\n".join(detected_objects))

            # Simulate real-time processing
            time.sleep(0.1)

if __name__ == "__main__":
    main()
