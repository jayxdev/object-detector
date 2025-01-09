import json

def save_json_output(detections, output_path):
    with open(output_path, 'w') as json_file:
        json.dump(detections, json_file, indent=4)
