import os
import json
import numpy as np
import cv2



def rotation_vector_to_matrix(rvec):
    """Convert rotation vector to rotation matrix."""
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    return rotation_matrix

def create_transform_matrix(rvec, tvec,colmap = False):
    """Create the 4x4 transformation matrix from rvec and tvec."""
    rotation_matrix = rotation_vector_to_matrix(rvec)
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = tvec.flatten()
    c2w = np.linalg.inv(transform_matrix)

    if(colmap == True):
        flip_mat = np.array([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ])
        out = np.matmul(c2w, flip_mat)

        return out
    
    
    c2w[0:3,2] *= -1 # flip the y and z axis
    c2w[0:3,1] *= -1
    c2w = c2w[[1,0,2,3],:]
    c2w[2,:] *= -1 # flip whole world upside down
    return c2w

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

def correct_to_center(rvec, tvec, cols =7,rows =5,square_size = 0.056):
    center_x = (cols - 1) * square_size / 2.0
    center_y = (rows - 1) * square_size / 2.0

    # Translation vector to move the origin to the center of the checkerboard
    T_center = np.array([[center_x], [center_y], [0]])

    # Convert rvec to rotation matrix R
    R, _ = cv2.Rodrigues(rvec)

    # Calculate new translation vector relative to the center
    tvec_center = np.dot(R, T_center) + tvec
    return rvec, tvec_center