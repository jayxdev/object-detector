import cv2
import torch
from collections import deque
import time
import numpy as np

# Initialize the YOLOv5 model
def initialize_model(model_path='models/yolov5s.pt'):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)  # Load YOLOv5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
    model.to(device)
    return model

# Function to detect objects in a video
def detect_objects(video_path, model=None, frame_skip=5, resize_factor=1, max_queue_size=10, confidence_threshold=0.3):
    if model is None:
        model = initialize_model()

    cap = cv2.VideoCapture(video_path)
    
    frame_queue = deque(maxlen=max_queue_size)
    detections = []
    mod_out = []  # Initialize empty list for model output
    frame_id = 0
    last_detection_time = time.time()  # Initialize to current time
    current_time=time.time()

    while cap.isOpened():
        framerate = cap.get(cv2.CAP_PROP_FPS)
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        # Resize frame for display
        resized_frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
        frame_queue.append(resized_frame)

        # Perform detection only on every 'detection_skip' frame
        if frame_id % frame_skip == 0 or frame_id == 1:
            results = model(resized_frame)
            mod_out = results.xyxy[0].cpu().numpy()
            last_detection_time = current_time
             # Calculate FPS safely
            current_time = time.time()
            time_diff = current_time - last_detection_time
            if time_diff > 0:
                fps = 1 / time_diff
            else:
                fps = 0  # Fallback FPS
        
        frame_detections = []

        # Draw bounding boxes from the last detection on the current frame
        for obj in mod_out:
            conf = obj[4].item() 
            if conf > confidence_threshold:
                # Extract bounding box and label
                xmin, ymin, xmax, ymax = map(int, obj[:4])
                class_id = int(obj[5].item())
                label = model.names[class_id]

                color = (0, 255, 0)
                cv2.rectangle(resized_frame, (xmin, ymin), (xmax, ymax), color, 1)  # Reduced border width to 1
                text = f'{label} {conf:.2f}'
                cv2.putText(resized_frame, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)  # Reduced text size to 0.3 and thickness to 1
                
                object_data = {
                    "object": label,  # Use the detected class name
                    "ID": class_id,  # Use row index as ID
                    "confidence": conf,
                    "bbox": [xmin,ymin,xmax, ymax]  # Update with the correct column names
                }
    
                # Example: Link a sub-object if detected (based on custom logic)
                # Example of linking sub-objects (adjust according to your project)
                if label == "person":
                    object_data["subobject"] = {
                        "object": "helmet",
                        "ID": 1,
                        "bbox": [100, 150, 200, 250]  # Dummy bounding box, replace with actual logic
                    }
    
                frame_detections.append(object_data)

        detections.append({
            "frame_id": frame_id,
            "detections": frame_detections
        })
        
        # Display FPS on the frame
        cv2.putText(resized_frame, f"Detection FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # Adjusted text size and thickness
        cv2.putText(resized_frame, f"Video FPS: {framerate:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        yield {
            "frame_id": frame_id,
            "detections": frame_detections,
            "frame": resized_frame
        }
        # Display the video
        cv2.imshow("Real-Time Object Detection", resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return detections
