import cv2
import numpy as np
from matplotlib import pyplot as plt

# ARUCO_DICT = cv2.aruco.DICT_ARUCO_ORIGINAL
# SQUARES_VERTICALLY = 10
# SQUARES_HORIZONTALLY = 7
# SQUARE_LENGTH = 0.03
# MARKER_LENGTH = 0.015

ARUCO_DICT = cv2.aruco.DICT_4X4_50
SQUARES_VERTICALLY = 7
SQUARES_HORIZONTALLY = 5
SQUARE_LENGTH = 0.056
MARKER_LENGTH = 0.042

def detect_pose(image, camera_matrix, dist_coeffs):
    # Undistort the image
    
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)
    
    # Define the aruco dictionary and charuco board
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    params = cv2.aruco.DetectorParameters()
    #cv2.imwrite("test.png",undistorted_image)
    # Detect markers in the undistorted image
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(undistorted_image, dictionary, parameters=params)
    
    # If at least one marker is detected
    if (marker_ids) is not None:
        # Interpolate CharUco corners
        charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, undistorted_image, board)
        
        # If enough corners are found, estimate the pose
        if charuco_retval:
            retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, None, None)

            # If pose estimation is successful, draw the axis
            if retval:
                cv2.drawFrameAxes(undistorted_image, camera_matrix, dist_coeffs, rvec, tvec, length=0.1, thickness=15)
                plt.imshow(undistorted_image)
        else:
            return False,None,None
        return charuco_retval,rvec,tvec
    return False,None,None


