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
from config.ConfigManager import ConfigManager
from calibration.json_utils.json_functions import generate_json_for_images

def calibrate_and_write(project_name, output_folder, config_manager):

    image_files = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith(".jpg")]
    image_files.sort()  # Ensure files are in order

    allCorners, allIds, imsize, num_of_detected_markers = read_chessboards(image_files)

    ret, mtx, dist, rvecs, tvecs = calibrate_camera(allCorners, allIds, imsize)

    if ret > 0:
        np.save(os.path.join(output_folder, "camera_matrix.npy"), mtx)
        np.save(os.path.join(output_folder, "camera_dist_coeff.npy"), dist)
        config_manager.update_project(project_name, {"calibrated": True, "camera_matrix": mtx.tolist(),
                                                     "dist_coeff": dist.tolist()})
    else:
        print("No calibration data were found")

    return mtx, dist

def main():
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
    config_file = args.config_file

    config_manager = ConfigManager(config_file)

    print(os.path.isdir(input_path))

    config_manager.update_project(project_name, {"images": output_folder})
    mtx, dist = calibrate_and_write(project_name, output_folder, config_manager)

    # if os.path.isdir(input_path):
        
    # else:
    #     err = os.system(f"sfextract --frame-count {num_of_images} {input_path} --output {output_folder}")  
    #     if err:
    #         print("Error occurred while generating images from video")
    #         sys.exit(1)
    #     else:
    #         config_manager.update_project(project_name, {"images": output_folder})
    #         mtx, dist = calibrate_and_write(project_name, output_folder, config_manager)

    generate_json_for_images(output_folder, os.path.join(output_folder, "transforms_centered.json"), mtx, dist, colmap=True)

if __name__ == "__main__":
    main()
