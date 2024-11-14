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


def do_system(arg):
	print(f"==== running: {arg}", flush=True)
	err = os.system(arg)
	if err:
		print("FATAL: command failed", flush=True)
		sys.exit(err)

config_manager = ConfigManagerAPI("http://127.0.0.1:5001")

ARUCO_DICT = cv2.aruco.DICT_6X6_250
SQUARES_VERTICALLY = 13
SQUARES_HORIZONTALLY = 9
SQUARE_LENGTH = 0.03
MARKER_LENGTH = 0.02

dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
print("Setup with this board", board.getDictionary())


parser = argparse.ArgumentParser(description="Camera calibration and image processing script.")
parser.add_argument('--project_name', type=str, required=True, help='Name of the project')
parser.add_argument('--input', type=str, required=True, help='Input folder or video file path')

parser.add_argument('--mtx', nargs=9,type = float, default = None, required=True, help='Path to the calibration camera matrix data ')
parser.add_argument('--dist',  nargs=14,type = float, default = None, required=True,help='Path to the calibration camera distortion coefficients')

parser.add_argument("--remove_background", action="store_true", help="Perform background removal before acquiring poses. Model to use specify in bg_model")
parser.add_argument("--bg_model", default="isnet-general-use",choices=["sam", "isnet-general-use","u2net","birefnet-general","birefnet-general-lite"], help="Model that will be used for removing background. Sam(object point in center of image) or isnet-general")

args = parser.parse_args()

project_name = args.project_name
input_path = args.input
append_rembg = False

if(args.remove_background):
    append_rembg = True

if args.remove_background:
		do_system(f"python ../scripts/backremover.py --model {args.bg_model} --input_dir {input_path}")

mtx  = np.array(args.mtx).reshape(3, 3)

dist = np.array(args.dist)

out_path = os.path.join(input_path, "transforms_centered.json")

print("Starting pose generation ...",flush = True)
generate_json_for_images(input_path, out_path, mtx, dist, board = board ,scale = 3,append_rembg = append_rembg,colmap=True)
config_manager.update_project(project_name,{"transforms_path":out_path})

camera_with_frustums(out_path, os.path.join(input_path, "cameras.html"))


