import cv2
from src.object_detection import detect_objects
from src.json_output import save_json_output

# Load the video or image
video_path = "data/traffic.mp4"  # Path to your sample video

# Run the detection
detections = detect_objects(video_path)

# Save the detection results as JSON
save_json_output(detections, "output/detection_results.json")
