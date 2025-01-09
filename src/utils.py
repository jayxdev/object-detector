import cv2
import os
import numpy as np

# Function to save cropped sub-object images
def save_subobject_image(frame, bbox, object_id, subobject_name, output_folder="output"):
    x1, y1, x2, y2 = bbox
    sub_object_image = frame[y1:y2, x1:x2]
    
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Save the image
    file_name = f"{subobject_name}_{object_id}.jpg"
    output_path = os.path.join(output_folder, file_name)
    cv2.imwrite(output_path, sub_object_image)
