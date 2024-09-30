import json
import cv2
import os
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join('..')))


from calibration.geometry_utils.geometry_functions import create_transform_matrix
from calibration.geometry_utils.geometry_functions import correct_to_center
from calibration.geometry_utils.geometry_functions import closest_point_2_lines
from calibration.geometry_utils.geometry_functions import rotmat

from calibration.cv2api.detect import detect_pose

def undistort_image(image, camera_matrix, dist_coeffs):
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    return undistorted_image


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
        "h": template.get("h"),
        "aabb_scale": template.get("aabb_scale"),
        "scale": template.get("scale")
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


def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()

def sharpness(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = variance_of_laplacian(gray)
	return fm


def compute_scaling_factor(data_json):
    # Load the transforms.json file
    
    camera_positions = []

    for frame in data_json['frames']:
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

    print("Bounding box minimum coordinates:", min_coords)
    print("Bounding box maximum coordinates:", max_coords)
    print("Scene dimensions (x, y, z):", dimensions)
    print("Maximum dimension:", max_dimension)
    print("Scaling factor S =", scaling_factor)

    return scaling_factor


def generate_json_for_images(folder_path,output_json_path , camera_matrix,dist_coeff,scale = 3,colmap = False):
    """Generate JSON for a folder of images with given rvecs and tvecs."""

    # Example usage:
    # camera_matrix = np.load(folder_path+"camera_matrix.npy")
    # dist_coeff = np.load(folder_path+"camera_dist_coeff.npy")
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".jpg") or f.endswith(".png")or f.endswith(".bmp")]
    image = cv2.imread(image_files[0])
    #print(image)
    camera_params = {
        "camera_angle_x": 2 * np.arctan(image.shape[1] / (2 * camera_matrix[0,0])),
        "camera_angle_y": 2 * np.arctan(image.shape[0] / (2 * camera_matrix[1,1])),
        "fl_x": camera_matrix[0,0],
        "fl_y": camera_matrix[1,1],
        "k1": dist_coeff[0],
        "k2": dist_coeff[1],
        "k3": dist_coeff[4],
        "k4": 0,
        "p1": dist_coeff[2],
        "p2": dist_coeff[3],
        "is_fisheye": False,
        "cy": camera_matrix[1,2],
        "cx": camera_matrix[0,2],
        "w": image.shape[1],
        "h": image.shape[0],
        "aabb_scale": 1,
        "scale": scale,
        "poses_from": "charuco"
    }

    data = {
        "camera_angle_x": camera_params["camera_angle_x"],
        "camera_angle_y": camera_params["camera_angle_y"],
        "fl_x": camera_params["fl_x"],
        "fl_y": camera_params["fl_y"],
        "k1": camera_params["k1"],
        "k2": camera_params["k2"],
        "k3": camera_params["k3"],
        "k4": camera_params["k4"],
        "p1": camera_params["p1"],
        "p2": camera_params["p2"],
        "is_fisheye": camera_params["is_fisheye"],
        "cx": camera_params["cx"],
        "cy": camera_params["cy"],
        "w": camera_params["w"],
        "h": camera_params["h"],
        "aabb_scale": camera_params["aabb_scale"],
        "scale": camera_params["scale"],
        "poses_from": camera_params["poses_from"],
        "frames": []
    }


    image_files = sorted(os.listdir(folder_path))
    print(image_files)
    up = np.array([0.0, 0.0, 0.0])
    for i, image_file in enumerate(image_files):
        if image_file.endswith(('.png', '.jpg', '.jpeg',".bmp")):
            image = cv2.imread(os.path.join(folder_path,image_file))
            
            retval, rvec, tvec = detect_pose(image, camera_matrix, dist_coeff)
            
            if rvec is not None and rvec.any():
                print(image_file,len(rvec))
            else:
                print(image_file, "No markers detected:/")
            #print("Found something?")

            if(retval > 0):
                #print(retval,rvec)
                try:
                    rvec,tvec = correct_to_center(rvec,tvec)
                
                    
                    c2w = create_transform_matrix(rvec, tvec)
                    up += c2w[0:3,1]
                    frame_data = {
                        "file_path": image_file,
                        "sharpness": sharpness(image),  # Placeholder, update as needed
                        "transform_matrix": c2w
                    }
                    data["frames"].append(frame_data)
                    #print("test")
                except Exception as e: 
                    print(e)
    
    #print(data["frames"])

    nframes = len(data["frames"])
    #print(nframes)
    if(colmap == False):
        up = up / np.linalg.norm(up)
        print("up vector was", up)
        R = rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
        R = np.pad(R,[0,1])
        R[-1, -1] = 1

        for f in data["frames"]:
            f["transform_matrix"] = np.matmul(R, f["transform_matrix"]) # rotate up to be the z axis

        # find a central point they are all looking at
        print("computing center of attention...")
        totw = 0.0
        totp = np.array([0.0, 0.0, 0.0])
        for f in data["frames"]:
            mf = f["transform_matrix"][0:3,:]
            for g in data["frames"]:
                mg = g["transform_matrix"][0:3,:]
                p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
                if w > 0.00001:
                    totp += p*w
                    totw += w
        if totw > 0.0:
            totp /= totw
        print(totp) # the cameras are looking at totp
        for f in data["frames"]:
            f["transform_matrix"][0:3,3] -= totp

        avglen = 0.
        for f in data["frames"]:
            avglen += np.linalg.norm(f["transform_matrix"][0:3,3])
        avglen /= nframes
        print("avg camera distance from origin", avglen)
        for f in data["frames"]:
            f["transform_matrix"][0:3,3] *= 4.0 / avglen # scale to "nerf sized"

    for f in data["frames"]:
        f["transform_matrix"] = f["transform_matrix"].tolist()

    scaling_factor = compute_scaling_factor(data)
    
    data["computed_scaling_factor"] = scaling_factor

    with open(output_json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

# def vizualize_camera(transforms_json):
    