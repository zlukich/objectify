#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
from glob import glob
import os
from pathlib import Path, PurePosixPath

import numpy as np
import json
import sys
import math
import cv2
import os
import shutil


import json
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio


import sqlite3  

sys.path.append(os.path.abspath(os.path.join('..','..', 'objectify','lib')))

from config.ConfigManagerServer import ConfigManagerAPI


config_manager = ConfigManagerAPI("http://127.0.0.1:5001")


def image_ids_to_pair_id(image_id1, image_id2):
    min_id = min(image_id1, image_id2)
    max_id = max(image_id1, image_id2)
    return int(min_id * 2147483647 + max_id)


# Add this function to visualize matches
def visualize_matches(args):
    db_path = os.path.join(args.images, args.colmap_db)
    if not os.path.exists(db_path):
        print(f"COLMAP database not found at {db_path}")
        return

    if args.remove_background:
        images_path = os.path.join(args.images, "rembg")
    else:
        images_path = args.images

    matches_dir = os.path.join(args.images, 'matches')
    if not os.path.exists(matches_dir):
        os.makedirs(matches_dir)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get image_id to image_name mapping
    cursor.execute("SELECT image_id, name FROM images")
    image_id_to_name = {row[0]: row[1] for row in cursor.fetchall()}
    image_name_to_id = {v: k for k, v in image_id_to_name.items()}

    # For each specified image pair
    for pair_str in args.image_pairs:
        img_name1, img_name2 = pair_str.split(',')

        if img_name1 not in image_name_to_id or img_name2 not in image_name_to_id:
            print(f"One of the images {img_name1} or {img_name2} not found in database")
            continue

        image_id1 = image_name_to_id[img_name1]
        image_id2 = image_name_to_id[img_name2]

        # Compute pair_id
        pair_id = image_ids_to_pair_id(image_id1, image_id2)

        # Get matches
        cursor.execute("SELECT rows, cols, data FROM matches WHERE pair_id = ?", (pair_id,))
        result = cursor.fetchone()
        if result is None:
            print(f"No matches found between {img_name1} and {img_name2}")
            continue

        rows, cols, data = result
        matches = np.frombuffer(data, dtype=np.uint32).reshape((rows, cols))

        # Get keypoints for both images
        cursor.execute("SELECT rows, cols, data FROM keypoints WHERE image_id = ?", (image_id1,))
        result1 = cursor.fetchone()
        if result1 is None:
            print(f"No keypoints found for image {img_name1}")
            continue
        rows1, cols1, data1 = result1
        keypoints1 = np.frombuffer(data1, dtype=np.float32).reshape((rows1, cols1))

        cursor.execute("SELECT rows, cols, data FROM keypoints WHERE image_id = ?", (image_id2,))
        result2 = cursor.fetchone()
        if result2 is None:
            print(f"No keypoints found for image {img_name2}")
            continue
        rows2, cols2, data2 = result2
        keypoints2 = np.frombuffer(data2, dtype=np.float32).reshape((rows2, cols2))

        # Load images
        image_path1 = os.path.join(images_path, img_name1)
        image_path2 = os.path.join(images_path, img_name2)

        img1 = cv2.imread(image_path1)
        img2 = cv2.imread(image_path2)

        if img1 is None or img2 is None:
            print(f"One of the images {image_path1} or {image_path2} not found")
            continue

        # Convert keypoints to cv2.KeyPoint format using positional arguments
        kp1 = [cv2.KeyPoint(pt[0], pt[1], pt[2] if cols1 >= 3 else 1) for pt in keypoints1]
        kp2 = [cv2.KeyPoint(pt[0], pt[1], pt[2] if cols2 >= 3 else 1) for pt in keypoints2]

        # Convert matches to DMatch format
        matches_cv = [cv2.DMatch(_queryIdx=m[0], _trainIdx=m[1], _distance=0) for m in matches]

        # Draw matches
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches_cv, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Save image
        output_file = os.path.join(matches_dir, f"{os.path.splitext(img_name1)[0]}_{os.path.splitext(img_name2)[0]}_matches.png")
        cv2.imwrite(output_file, img_matches)
        print(f"Saved matches visualization to {output_file}")

    conn.close()

def apply_alpha_mask_and_save(args):
    """
    Apply the alpha mask to images and save them with the background removed (black).
    """
    if args.remove_background:
        images_path = os.path.join(args.images, "rembg")
    else:
        images_path = args.images

    masked_images_path = os.path.join(args.images, 'masked_images')
    if not os.path.exists(masked_images_path):
        os.makedirs(masked_images_path)

    # Get all images
    for image_file in glob(os.path.join(images_path, '*.png')):  # Assuming images are in PNG format with alpha
        image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)  # Load image with alpha channel
        if image is None or image.shape[2] != 4:
            print(f"Skipping non-alpha image: {image_file}")
            continue

        # Extract alpha channel
        alpha_channel = image[:, :, 3]
        mask = (alpha_channel > 0).astype(np.uint8)

        # Apply mask: Set background (alpha=0) pixels to black in the RGB channels
        masked_image = cv2.bitwise_and(image[:, :, :3], image[:, :, :3], mask=mask)

        # Save the masked image (only RGB channels)
        output_file = os.path.join(masked_images_path, os.path.basename(image_file))
        cv2.imwrite(output_file, masked_image)
        print(f"Saved masked image to {output_file}")
    
    return masked_images_path

def visualize_keypoints(args):
    db_path = os.path.join(args.images, args.colmap_db)
    if not os.path.exists(db_path):
        print(f"COLMAP database not found at {db_path}")
        return

    if args.remove_background:
        images_path = os.path.join(args.images, "rembg")
    else:
        images_path = args.images

    keypoints_dir = os.path.join(args.images, 'keypoints')
    if not os.path.exists(keypoints_dir):
        os.makedirs(keypoints_dir)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get image_id to image_name mapping
    cursor.execute("SELECT image_id, name FROM images")
    image_id_to_name = {row[0]: row[1] for row in cursor.fetchall()}

    # For each image
    for image_id, image_name in image_id_to_name.items():
        # Get keypoints
        cursor.execute("SELECT rows, cols, data FROM keypoints WHERE image_id = ?", (image_id,))
        result = cursor.fetchone()
        if result is None:
            continue
        rows, cols, data = result
        # Convert data to numpy array
        keypoints = np.frombuffer(data, dtype=np.float32).reshape((rows, cols))
        # x, y coordinates
        x = keypoints[:, 0]
        y = keypoints[:, 1]

        # Load image
        image_path = os.path.join(images_path, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Image {image_path} not found")
            continue

        # Draw keypoints
        for xi, yi in zip(x, y):
            cv2.circle(image, (int(round(xi)), int(round(yi))), 2, (0, 255, 0), -1)

        # Save image
        output_path = os.path.join(keypoints_dir, image_name)
        cv2.imwrite(output_path, image)
        print(f"Saved keypoints visualization to {output_path}")

    conn.close()


### Visualization specific functions TODO change it to import function /lib/calibration/viz_utils/viz_cameras
def get_frustum_vertices(scale=0.1):
    """Define frustum vertices in camera coordinates."""
    return np.array([
        [0, 0, 0],  # Camera center
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1]
    ]) * scale

def transform_frustum(vertices, transform_matrix):
    """Transform frustum vertices to world coordinates."""
    rotation_matrix = transform_matrix[:3, :3]
    camera_position = transform_matrix[:3, 3]
    vertices_world = (rotation_matrix @ vertices.T).T + camera_position
    return vertices_world

def camera_with_frustums(json_path, write_image_path = "cameras.png"):

    # Load JSON file containing transformation matrices
    with open(json_path, 'r') as f:
        data = json.load(f)


    # Extract frames from JSON data
    frames = data['frames']
    num_cameras = len(frames)  # Count the number of cameras
    # Plot camera frustums
    fig = go.Figure()

    for frame in frames:
        transform_matrix = np.array(frame['transform_matrix'])
        frustum_vertices = get_frustum_vertices()
        frustum_vertices_world = transform_frustum(frustum_vertices, transform_matrix)
        
        # Plot camera center
        fig.add_trace(go.Scatter3d(
            x=[frustum_vertices_world[0, 0]], 
            y=[frustum_vertices_world[0, 1]], 
            z=[frustum_vertices_world[0, 2]],
            mode='markers',
            marker=dict(size=3, color='red'),
            name='Camera Center'
        ))
        
        # Plot frustum edges
        edges = [
            (0, 1), (0, 2), (0, 3), (0, 4),  # From camera center to frustum vertices
            (1, 2), (2, 3), (3, 4), (4, 1)   # Between frustum vertices
        ]
        
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=[frustum_vertices_world[edge[0], 0], frustum_vertices_world[edge[1], 0]],
                y=[frustum_vertices_world[edge[0], 1], frustum_vertices_world[edge[1], 1]],
                z=[frustum_vertices_world[edge[0], 2], frustum_vertices_world[edge[1], 2]],
                mode='lines',
                line=dict(color='red'),
                name='Frustum Edge'
            ))

    # Set plot labels
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title='Camera Frustums'
    )
    # Optionally, add an annotation for the number of cameras
    fig.add_annotation(
        text=f'Number of Cameras: {num_cameras}',
        xref='paper', yref='paper',
        x=0.5, y=1.05, showarrow=False,
        font=dict(size=14),
        align='center'
    )
    # Show plot
    #pio.show(fig)
    fig.write_html(write_image_path)


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SCRIPTS_FOLDER = os.path.join(ROOT_DIR, "scripts")

def parse_args():
	parser = argparse.ArgumentParser(description="Convert a text colmap export to nerf format transforms.json; optionally convert video to images, and optionally run colmap in the first place.")

	parser.add_argument("--video_in", default="", help="Run ffmpeg first to convert a provided video file into a set of images. Uses the video_fps parameter also.")
	parser.add_argument("--video_fps", default=2)
	parser.add_argument("--time_slice", default="", help="Time (in seconds) in the format t1,t2 within which the images should be generated from the video. E.g.: \"--time_slice '10,300'\" will generate images only from 10th second to 300th second of the video.")
	parser.add_argument("--run_colmap", action="store_true", help="run colmap first on the image folder")
	parser.add_argument("--colmap_matcher", default="sequential", choices=["exhaustive","sequential","spatial","transitive","vocab_tree"], help="Select which matcher colmap should use. Sequential for videos, exhaustive for ad-hoc images.")
	parser.add_argument("--colmap_db", default="colmap.db", help="colmap database filename")
	parser.add_argument("--colmap_camera_model", default="OPENCV", choices=["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "OPENCV_FISHEYE"], help="Camera model")
	parser.add_argument("--colmap_camera_params", default="", help="Intrinsic parameterscolmap_matcher the chosen model. Format: fx,fy,cx,cy,dist")
	parser.add_argument("--images", default="images", help="Input path to the images.")
	parser.add_argument("--text", default="colmap_text", help="Input path to the colmap text files (set automatically if --run_colmap is used).")
	parser.add_argument("--aabb_scale", default=1, choices=["1", "2", "4", "8", "16", "32", "64", "128"], help="Large scene scale factor. 1=scene fits in unit cube; power of 2 up to 128")
	parser.add_argument("--skip_early", default=0, help="Skip this many images from the start.")
	parser.add_argument("--keep_colmap_coords", action="store_true", help="Keep transforms.json in COLMAP's original frame of reference (this will avoid reorienting and repositioning the scene for preview and rendering).")
	parser.add_argument("--out", default="transforms_centered.json", help="Output JSON file path.")
	parser.add_argument("--vocab_path", default="", help="Vocabulary tree path.")
	parser.add_argument("--overwrite", action="store_true", help="Do not ask for confirmation for overwriting existing images and COLMAP data.")
	parser.add_argument("--mask_categories", nargs="*", type=str, default=[], help="Object categories that should be masked out from the training images. See `scripts/category2id.json` for supported categories.")
 
	# Added arguments in objectify project
	parser.add_argument("--remove_background", action="store_true", help="Perform background removal before acquiring poses. Model to use specify in bg_model")
	parser.add_argument("--bg_model", default="isnet-general-use",choices=["sam", "isnet-general-use","u2net","birefnet-general","birefnet-general-lite"], help="Model that will be used for removing background. Sam(object point in center of image) or isnet-general")
	parser.add_argument("--single_camera", default=1,choices=['0', '1'], help="Option that provides to colmap information that camera with constant flx,fly was used")
	parser.add_argument(
			"--image_pairs",
			nargs="*",
			default=[],
			help="Pairs of images to visualize matches for. Each pair is specified as 'image1.jpg,image2.jpg'"
		)

	args = parser.parse_args()
	return args

def do_system(arg):
	print(f"==== running: {arg}", flush=True)
	err = os.system(arg)
	if err:
		print("FATAL: command failed", flush=True)
		sys.exit(err)

def run_ffmpeg(args):
	ffmpeg_binary = "ffmpeg"

	# On Windows, if FFmpeg isn't found, try automatically downloading it from the internet
	if os.name == "nt" and os.system(f"where {ffmpeg_binary} >nul 2>nul") != 0:
		ffmpeg_glob = os.path.join(ROOT_DIR, "external", "ffmpeg", "*", "bin", "ffmpeg.exe")
		candidates = glob(ffmpeg_glob)
		if not candidates:
			print("FFmpeg not found. Attempting to download FFmpeg from the internet.")
			do_system(os.path.join(SCRIPTS_FOLDER, "download_ffmpeg.bat"))
			candidates = glob(ffmpeg_glob)

		if candidates:
			ffmpeg_binary = candidates[0]

	if not os.path.isabs(args.images):
		args.images = os.path.join(os.path.dirname(args.video_in), args.images)

 
	images = "\"" + args.images + "\""
	video =  "\"" + args.video_in + "\""
	fps = float(args.video_fps) or 1.0
	print(f"running ffmpeg with input video file={video}, output image folder={images}, fps={fps}.", flush=True)
	# if not args.overwrite and (input(f"warning! folder '{images}' will be deleted/replaced. continue? (Y/n)").lower().strip()+"y")[:1] != "y":
	# 	sys.exit(1)
	try:
		# Passing Images' Path Without Double Quotes
		shutil.rmtree(args.images)
	except:
		pass
	do_system(f"mkdir {images}")

	time_slice_value = ""
	time_slice = args.time_slice
	if time_slice:
		start, end = time_slice.split(",")
		time_slice_value = f",select='between(t\,{start}\,{end})'"
	do_system(f"{ffmpeg_binary} -i {video} -qscale:v 1 -qmin 1 -vf \"fps={fps}{time_slice_value}\" {images}/%04d.jpg")

def run_colmap(args):
	colmap_binary = "colmap"

	# On Windows, if FFmpeg isn't found, try automatically downloading it from the internet
	if os.name == "nt" and os.system(f"where {colmap_binary} >nul 2>nul") != 0:
		colmap_glob = os.path.join(ROOT_DIR, "external", "colmap", "*", "COLMAP.bat")
		candidates = glob(colmap_glob)
		if not candidates:
			print("COLMAP not found. Attempting to download COLMAP from the internet.")
			do_system(os.path.join(SCRIPTS_FOLDER, "download_colmap.bat"))
			candidates = glob(colmap_glob)

		if candidates:
			colmap_binary = candidates[0]
 
	project_name = os.path.basename(args.images)
 
	
	conf = config_manager.get_project(project_name)
 
	
	if args.remove_background:
		do_system(f"python ../scripts/backremover.py --model {args.bg_model} --input_dir {args.images}")


	
	if(args.remove_background):
		images = "\"" + os.path.join(args.images,"rembg") + "\""
		#print("I AM HERE")
	else:
		images = "\"" + args.images + "\""
	#images = "\"" + args.images + "\""

	
	db = os.path.join(args.images,args.colmap_db)
	print(images)
	db_noext=str(Path(db).with_suffix(""))

	if args.text=="text":
		args.text=db_noext+"_text"
	text=os.path.join(args.images,args.text)
	sparse=db_noext+"_sparse"
	print(f"running colmap with:\n\tdb={db}\n\timages={images}\n\tsparse={sparse}\n\ttext={text}", flush=True)
	# if not args.overwrite and (input(f"warning! folders '{sparse}' and '{text}' will be deleted/replaced. continue? (Y/n)").lower().strip()+"y")[:1] != "y":
	# 	sys.exit(1)
	if os.path.exists(db):
		os.remove(db)
  
	do_system(f"{colmap_binary} feature_extractor --log_to_stderr 1 --ImageReader.camera_model {args.colmap_camera_model} --ImageReader.camera_params \"{args.colmap_camera_params}\" --SiftExtraction.estimate_affine_shape=true --SiftExtraction.domain_size_pooling=true --ImageReader.single_camera {args.single_camera} --database_path {db} --image_path {images} --SiftExtraction.max_num_features 8128 --SiftExtraction.use_gpu 1")
	match_cmd = f"{colmap_binary} {args.colmap_matcher}_matcher --SiftMatching.guided_matching=true --database_path {db} --SiftMatching.use_gpu 1"
	if args.vocab_path:
		match_cmd += f" --VocabTreeMatching.vocab_tree_path {args.vocab_path}"
	do_system(match_cmd)
	try:
		shutil.rmtree(sparse)
	except:
		pass
	do_system(f"mkdir {sparse}")
	do_system(f"{colmap_binary} mapper --database_path {db} --image_path {images} --output_path {sparse} --Mapper.ba_use_gpu 1")
	do_system(f"{colmap_binary} bundle_adjuster --input_path {sparse}/0 --output_path {sparse}/0 --BundleAdjustment.refine_principal_point 1 --BundleAdjustment.max_num_iterations 100 --BundleAdjustment.use_gpu 1")
	try:
		shutil.rmtree(text)
	except:
		pass
	do_system(f"mkdir {text}")
	do_system(f"{colmap_binary} model_converter --input_path {sparse}/0 --output_path {text} --output_type TXT")
 
#  # Dense reconstruction
# 	dense = db_noext + "_dense"
# 	dense_model = os.path.join(dense, "fused.ply")
# 	try:
# 		shutil.rmtree(dense)
# 	except:
# 		pass
# 	do_system(f"mkdir {dense}")
# 	# Copy sparse model to the workspace
# 	sparse_model_path = os.path.join(sparse, "0")  # Path to the sparse model
# 	workspace_sparse_path = os.path.join(dense, "sparse")  # Target sparse directory in dense workspace
# 	shutil.copytree(sparse_model_path, workspace_sparse_path)

	# # Create the stereo directory
	# stereo_path = os.path.join(dense, "stereo")
	# do_system(f"mkdir {stereo_path}")

	# # Patch Match Stereo (compute depth maps)
	# do_system(
	# 	f"{colmap_binary} patch_match_stereo "
	# 	f"--workspace_path {dense} "
	# 	f"--workspace_format COLMAP "
	# 	f"--PatchMatchStereo.depth_min 0.5 "
	# 	f"--PatchMatchStereo.depth_max 5.0 "
	# 	f"--PatchMatchStereo.gpu_index 0"
	# )

	# # Stereo Fusion (fuse depth maps into a dense point cloud)
	# dense_model = os.path.join(dense, "fused.ply")
	# do_system(
	# 	f"{colmap_binary} stereo_fusion "
	# 	f"--workspace_path {dense} "
	# 	f"--workspace_format COLMAP "
	# 	f"--input_type geometric "
	# 	f"--output_type PLY "
	# 	f"--output_path {dense_model} "
	# 	f"--StereoFusion.use_cache 1"
	# )

	# # Poisson Reconstruction (create a dense mesh)
	# poisson_model = os.path.join(dense, "poisson.ply")
	# do_system(
	# 	f"{colmap_binary} poisson_mesher "
	# 	f"--input_path {dense_model} "
	# 	f"--output_path {poisson_model}"
	# )

# print(f"Dense reconstruction completed. Poisson mesh saved at {poisson_model}")

def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()

def sharpness(imagePath):
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = variance_of_laplacian(gray)
	return fm

def qvec2rotmat(qvec):
	return np.array([
		[
			1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
			2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
			2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
		], [
			2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
			1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
			2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
		], [
			2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
			2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
			1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
		]
	])

def rotmat(a, b):
	a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
	v = np.cross(a, b)
	c = np.dot(a, b)
	# handle exception for the opposite direction input
	if c < -1 + 1e-10:
		return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
	da = da / np.linalg.norm(da)
	db = db / np.linalg.norm(db)
	c = np.cross(da, db)
	denom = np.linalg.norm(c)**2
	t = ob - oa
	ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
	tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
	if ta > 0:
		ta = 0
	if tb > 0:
		tb = 0
	return (oa+ta*da+ob+tb*db) * 0.5, denom

if __name__ == "__main__":
	args = parse_args()
 
 
	
	
	if args.video_in != "":
		run_ffmpeg(args)
	if args.run_colmap:
		run_colmap(args)

	#visualize_keypoints(args)
	#visualize_matches(args)
	AABB_SCALE = int(args.aabb_scale)
	SKIP_EARLY = int(args.skip_early)
	if(args.remove_background):
		IMAGE_FOLDER = os.path.join(args.images,"rembg")
	else:
		IMAGE_FOLDER = args.images
	TEXT_FOLDER = os.path.join(args.images,args.text)
	OUT_PATH = os.path.join(args.images,args.out)
 

	# Check that we can save the output before we do a lot of work
	try:
		open(OUT_PATH, "a").close()
	except Exception as e:
		print(f"Could not save transforms JSON to {OUT_PATH}: {e}", flush=True)
		sys.exit(1)

	print(f"outputting to {OUT_PATH}...", flush=True)
	cameras = {}
	with open(os.path.join(TEXT_FOLDER,"cameras.txt"), "r") as f:
		camera_angle_x = math.pi / 2
		for line in f:
			# 1 SIMPLE_RADIAL 2048 1536 1580.46 1024 768 0.0045691
			# 1 OPENCV 3840 2160 3178.27 3182.09 1920 1080 0.159668 -0.231286 -0.00123982 0.00272224
			# 1 RADIAL 1920 1080 1665.1 960 540 0.0672856 -0.0761443
			if line[0] == "#":
				continue
			els = line.split(" ")
			camera = {}
			camera_id = int(els[0])
			camera["w"] = float(els[2])
			camera["h"] = float(els[3])
			camera["fl_x"] = float(els[4])
			camera["fl_y"] = float(els[4])
			camera["k1"] = 0
			camera["k2"] = 0
			camera["k3"] = 0
			camera["k4"] = 0
			camera["p1"] = 0
			camera["p2"] = 0
			camera["cx"] = camera["w"] / 2
			camera["cy"] = camera["h"] / 2
			camera["is_fisheye"] = False
			if els[1] == "SIMPLE_PINHOLE":
				camera["cx"] = float(els[5])
				camera["cy"] = float(els[6])
			elif els[1] == "PINHOLE":
				camera["fl_y"] = float(els[5])
				camera["cx"] = float(els[6])
				camera["cy"] = float(els[7])
			elif els[1] == "SIMPLE_RADIAL":
				camera["cx"] = float(els[5])
				camera["cy"] = float(els[6])
				camera["k1"] = float(els[7])
			elif els[1] == "RADIAL":
				camera["cx"] = float(els[5])
				camera["cy"] = float(els[6])
				camera["k1"] = float(els[7])
				camera["k2"] = float(els[8])
			elif els[1] == "OPENCV":
				camera["fl_y"] = float(els[5])
				camera["cx"] = float(els[6])
				camera["cy"] = float(els[7])
				camera["k1"] = float(els[8])
				camera["k2"] = float(els[9])
				camera["p1"] = float(els[10])
				camera["p2"] = float(els[11])
			elif els[1] == "SIMPLE_RADIAL_FISHEYE":
				camera["is_fisheye"] = True
				camera["cx"] = float(els[5])
				camera["cy"] = float(els[6])
				camera["k1"] = float(els[7])
			elif els[1] == "RADIAL_FISHEYE":
				camera["is_fisheye"] = True
				camera["cx"] = float(els[5])
				camera["cy"] = float(els[6])
				camera["k1"] = float(els[7])
				camera["k2"] = float(els[8])
			elif els[1] == "OPENCV_FISHEYE":
				camera["is_fisheye"] = True
				camera["fl_y"] = float(els[5])
				camera["cx"] = float(els[6])
				camera["cy"] = float(els[7])
				camera["k1"] = float(els[8])
				camera["k2"] = float(els[9])
				camera["k3"] = float(els[10])
				camera["k4"] = float(els[11])
			else:
				print("Unknown camera model ", els[1])
			# fl = 0.5 * w / tan(0.5 * angle_x);
			camera["camera_angle_x"] = math.atan(camera["w"] / (camera["fl_x"] * 2)) * 2
			camera["camera_angle_y"] = math.atan(camera["h"] / (camera["fl_y"] * 2)) * 2
			camera["fovx"] = camera["camera_angle_x"] * 180 / math.pi
			camera["fovy"] = camera["camera_angle_y"] * 180 / math.pi

			print(f"camera {camera_id}:\n\tres={camera['w'],camera['h']}\n\tcenter={camera['cx'],camera['cy']}\n\tfocal={camera['fl_x'],camera['fl_y']}\n\tfov={camera['fovx'],camera['fovy']}\n\tk={camera['k1'],camera['k2']} p={camera['p1'],camera['p2']} ", flush=True)
			cameras[camera_id] = camera

	if len(cameras) == 0:
		print("No cameras found!", flush=True)
		sys.exit(1)

	with open(os.path.join(TEXT_FOLDER,"images.txt"), "r") as f:
		i = 0
		bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
		if len(cameras) == 1:
			camera = cameras[camera_id]
			out = {
				"camera_angle_x": camera["camera_angle_x"],
				"camera_angle_y": camera["camera_angle_y"],
				"fl_x": camera["fl_x"],
				"fl_y": camera["fl_y"],
				"k1": camera["k1"],
				"k2": camera["k2"],
				"k3": camera["k3"],
				"k4": camera["k4"],
				"p1": camera["p1"],
				"p2": camera["p2"],
				"is_fisheye": camera["is_fisheye"],
				"cx": camera["cx"],
				"cy": camera["cy"],
				"w": camera["w"],
				"h": camera["h"],
				"aabb_scale": AABB_SCALE,
				"poses_from": "colmap",
				"frames": [],
			}
		else:
			out = {
				"frames": [],
				"aabb_scale": AABB_SCALE,
    			"poses_from": "colmap"
			}

		up = np.zeros(3)
		for line in f:
			line = line.strip()
			if line[0] == "#":
				continue
			i = i + 1
			if i < SKIP_EARLY*2:
				continue
			if  i % 2 == 1:
				elems=line.split(" ") # 1-4 is quat, 5-7 is trans, 9ff is filename (9, if filename contains no spaces)
				#name = str(PurePosixPath(Path(IMAGE_FOLDER, elems[9])))
				# why is this requireing a relitive path while using ^
				image_rel = os.path.relpath(IMAGE_FOLDER)
				name = str(f"./{image_rel}/{'_'.join(elems[9:])}")
				b = sharpness(name)
				print(name, "sharpness=",b)
				image_id = int(elems[0])
				qvec = np.array(tuple(map(float, elems[1:5])))
				tvec = np.array(tuple(map(float, elems[5:8])))
				R = qvec2rotmat(-qvec)
				t = tvec.reshape([3,1])
				m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
				c2w = np.linalg.inv(m)
				if not args.keep_colmap_coords:
					c2w[0:3,2] *= -1 # flip the y and z axis
					c2w[0:3,1] *= -1
					c2w = c2w[[1,0,2,3],:]
					c2w[2,:] *= -1 # flip whole world upside down

					up += c2w[0:3,1]
				path, filename = os.path.split(name)
				if(args.remove_background):
					filename = os.path.join("rembg",filename) 
				frame = {"file_path":filename,"sharpness":b,"transform_matrix": c2w}
				if len(cameras) != 1:
					frame.update(cameras[int(elems[8])])
				out["frames"].append(frame)
	nframes = len(out["frames"])

	if args.keep_colmap_coords:
		flip_mat = np.array([
			[1, 0, 0, 0],
			[0, -1, 0, 0],
			[0, 0, -1, 0],
			[0, 0, 0, 1]
		])

		for f in out["frames"]:
			f["transform_matrix"] = np.matmul(f["transform_matrix"], flip_mat) # flip cameras (it just works)
	else:
		# don't keep colmap coords - reorient the scene to be easier to work with

		up = up / np.linalg.norm(up)
		print("up vector was", up, flush=True)
		R = rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
		R = np.pad(R,[0,1])
		R[-1, -1] = 1

		for f in out["frames"]:
			f["transform_matrix"] = np.matmul(R, f["transform_matrix"]) # rotate up to be the z axis

		# find a central point they are all looking at
		print("computing center of attention...", flush=True)
		totw = 0.0
		totp = np.array([0.0, 0.0, 0.0])
		for f in out["frames"]:
			mf = f["transform_matrix"][0:3,:]
			for g in out["frames"]:
				mg = g["transform_matrix"][0:3,:]
				p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
				if w > 0.00001:
					totp += p*w
					totw += w
		if totw > 0.0:
			totp /= totw
		print(totp) # the cameras are looking at totp
		for f in out["frames"]:
			f["transform_matrix"][0:3,3] -= totp

		avglen = 0.
		for f in out["frames"]:
			avglen += np.linalg.norm(f["transform_matrix"][0:3,3])
		avglen /= nframes
		print("avg camera distance from origin", avglen)
		for f in out["frames"]:
			f["transform_matrix"][0:3,3] *= 4.0 / avglen # scale to "nerf sized"

	for f in out["frames"]:
		f["transform_matrix"] = f["transform_matrix"].tolist()
	print(nframes,"frames", flush=True)
	print(f"writing {OUT_PATH}", flush=True)
 
	
	#compute scaling factor
	camera_positions = []
 
	for frame in out['frames']:
		transform_matrix = frame['transform_matrix']
		# Convert to numpy array
		transform_matrix = np.array(transform_matrix)
		# Get camera position in world coordinates
		# Multiply transform_matrix with [0, 0, 0, 1]^T
		camera_pos = np.dot(transform_matrix, np.array([0, 0, 0, 1]))
		# Take the x, y, z components
		camera_positions.append(camera_pos[:3])



	camera_positions = np.array(camera_positions)
	# Compute bounding box
	min_coords = np.min(camera_positions, axis=0)
	max_coords = np.max(camera_positions, axis=0)
	# Compute dimensions
	dimensions = max_coords - min_coords
	# Compute maximum dimension
	max_dimension = np.max(dimensions)
	# Compute scaling factor
	scaling_factor = 1 / max_dimension

	print("Bounding box minimum coordinates:", min_coords, flush=True)
	print("Bounding box maximum coordinates:", max_coords, flush=True)
	print("Scene dimensions (x, y, z):", dimensions, flush=True)
	print("Maximum dimension:", max_dimension, flush=True)
	print("Scaling factor S =", scaling_factor, flush=True)
 
	out["computed_scaling_factor"] = scaling_factor
	out["rembg"] = args.remove_background
 
	project_name = os.path.basename(args.images)
	  
	config_manager.update_project(project_name,{"rembg" : args.remove_background})
	config_manager.update_project(project_name,{"poses_from" : "colmap"})

	with open(OUT_PATH, "w") as outfile:
		json.dump(out, outfile, indent=2)

	camera_with_frustums(OUT_PATH, os.path.join(args.images,"cameras.html"))
 
	db = os.path.join(args.images,args.colmap_db)
 
	db_noext=str(Path(db).with_suffix(""))

	if args.text=="text":
		args.text=db_noext+"_text"
	text=os.path.join(args.images,args.text)
	sparse=db_noext+"_sparse"
 
	print("Removing colmap DB", flush=True)
	os.remove(db)
	print("Removing colmap text", flush=True)
	shutil.rmtree(text)
	# print("Removing colmap sparse", flush=True)
	# shutil.rmtree(sparse)

