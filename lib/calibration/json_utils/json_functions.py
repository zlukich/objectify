import json
from geometry_utils.geometry_functions import rotation_vector_to_matrix
from geometry_utils.geometry_functions import create_transform_matrix
from geometry_utils.geometry_functions import correct_to_center
from geometry_utils.geometry_functions import closest_point_2_lines
from geometry_utils.geometry_functions import rotmat
import cv2

def update_frames_with_camera_properties(template):
    # Extract the camera properties from the template
    camera_properties = {
        "camera_angle_x": template.get("camera_angle_x"),
        "camera_angle_y": template.get("camera_angle_y"),
        "fl_x": template.get("fl_x"),
        "fl_y": template.get("fl_y"),
        "k1": template.get("k1"),
        "k2": template.get("k2"),
        "k3": template.get("k3"),
        "k4": template.get("k4"),
        "p1": template.get("p1"),
        "p2": template.get("p2"),
        "is_fisheye": template.get("is_fisheye"),
        "cx": template.get("cx"),
        "cy": template.get("cy"),
        "w": template.get("w"),
        "h": template.get("h")
    }
    return camera_properties

def update_multi_camera_json(json_path):
    
    with open(json_path, 'r') as f:
        data = json.load(f)

    camera_properties = update_frames_with_camera_properties(data)
    out = {}
    for key,value in camera_properties.items():
        out["frames"] = data["frames"]
        for frame in out["frames"]:
            frame[key] = value
    return out

def combine_jsons(output_path, *json_paths):
    out = {'frames':[]}
    for json_path in json_paths:
        updated_json = update_multi_camera_json(json_path)
        out["frames"].extend(updated_json["frames"])

    with open(output_path, 'w') as json_file:
        json.dump(out, json_file, indent=4)
    return out


