import sys
import os
import numpy as np
from pathlib import Path
import argparse
import cv2
# Add the path to 'lib' directory
sys.path.append(os.path.abspath(os.path.join("..",'lib')))

# Import the necessary modules
from calibration.cv2api.calibrate import read_chessboards, calibrate_camera
from calibration.cv2api.detect import detect_pose
from config.ConfigManagerServer import ConfigManagerAPI
from calibration.json_utils.json_functions import generate_json_for_images

print("Starting a calibration", flush=True)

config_manager = ConfigManagerAPI("http://127.0.0.1:5001")

def calibrate_and_write(project_name, output_folder, config_manager, board):

    image_files = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".bmp")]
    image_files.sort()  # Ensure files are in order

    allCorners, allIds, imsize, num_of_detected_markers = read_chessboards(image_files, board)

    ret, mtx, dist, rvecs, tvecs = calibrate_camera(allCorners, allIds, imsize, board)

    
    if ret !=- 1:
        np.save(os.path.join(output_folder, "camera_matrix.npy"), mtx)
        np.save(os.path.join(output_folder, "camera_dist_coeff.npy"), dist)
        config_manager.update_project(project_name, {"calibrated": True, "camera_matrix": mtx.tolist(),
                                                     "dist_coeff": dist.tolist()})
        config_manager.update_current_work({"calibrated": True, "camera_matrix": mtx.tolist(),
                                                     "dist_coeff": dist.tolist()})
    else:
        print("No calibration data were found")
        return None, None

    return mtx, dist




parser = argparse.ArgumentParser(description="Camera calibration and image processing script.")
parser.add_argument('--project_name', type=str, required=True, help='Name of the project')
parser.add_argument('--input', type=str, required=True, help='Input folder or video file path')
parser.add_argument('--output_folder', type=str, required=True, help='Output folder to save images or results')
parser.add_argument('--config_file', type=str, required=True, help='Path to the configuration file')
parser.add_argument('--num_of_images', type=int, default=40, help='Number of images to extract from video if input is a video file')

args = parser.parse_args()

project_name = args.project_name
input_path = args.input
output_folder = args.output_folder
num_of_images = args.num_of_images

ARUCO_DICT = cv2.aruco.DICT_6X6_250
SQUARES_VERTICALLY = 13
SQUARES_HORIZONTALLY = 9
SQUARE_LENGTH = 0.03135
MARKER_LENGTH = 0.02105

dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
print("Setup with this board", board.getDictionary())

print(os.path.isdir(input_path))

config_manager.update_project(project_name, {"images": output_folder})
mtx, dist = calibrate_and_write(project_name, output_folder, config_manager, board = board)

if mtx is not None and mtx.any():
    config_manager.update_current_work({"camera_matrix": mtx.tolist(), "dist_coeff": dist.tolist()})
else:
    print("Calibration cannot be done, to little charuco markers detection", flush=True)
    raise Exception('calib failed')
# if os.path.isdir(input_path):
    
# else:
#     err = os.system(f"sfextract --frame-count {num_of_images} {input_path} --output {output_folder}")  
#     if err:
#         print("Error occurred while generating images from video")
#         sys.exit(1)
#     else:
#         config_manager.update_project(project_name, {"images": output_folder})
#         mtx, dist = calibrate_and_write(project_name, output_folder, config_manager)

#generate_json_for_images(output_folder, os.path.join(output_folder, "transforms_centered.json"), mtx, dist, colmap=True)


