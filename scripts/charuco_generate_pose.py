import sys
import os
import numpy as np
from pathlib import Path
import argparse

# Add the path to 'lib' directory
sys.path.append(os.path.abspath(os.path.join('..', 'lib')))

# Import the necessary modules
from calibration.cv2api.calibrate import read_chessboards, calibrate_camera
from calibration.cv2api.detect import detect_pose
from config.ConfigManagerServer import ConfigManagerAPI
from calibration.json_utils.json_functions import generate_json_for_images
from calibration.viz_utils.viz_cameras import camera_with_frustums
import cv2
config_manager = ConfigManagerAPI("http://127.0.0.1:5001")




parser = argparse.ArgumentParser(description="Camera calibration and image processing script.")
parser.add_argument('--project_name', type=str, required=True, help='Name of the project')
parser.add_argument('--input', type=str, required=True, help='Input folder or video file path')

parser.add_argument('--mtx', nargs=9,type = float, default = None, required=True, help='Path to the calibration camera matrix data ')
parser.add_argument('--dist',  nargs=14,type = float, default = None, required=True,help='Path to the calibration camera distortion coefficients')

args = parser.parse_args()

project_name = args.project_name
input_path = args.input

mtx  = np.array(args.mtx).reshape(3, 3)

dist = np.array(args.dist)

out_path = os.path.join(input_path, "transforms_centered.json")

print("Starting pose generation ...",flush = True)
generate_json_for_images(input_path, out_path, mtx, dist, scale = 3,colmap=True)
config_manager.update_project(project_name,{"transforms_path":out_path})

camera_with_frustums(out_path, os.path.join(input_path, "cameras.html"))


