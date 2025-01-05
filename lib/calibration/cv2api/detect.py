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
SQUARE_LENGTH = 0.03105
MARKER_LENGTH = 0.02105

# def detect_pose(image, camera_matrix, dist_coeffs):
#     # Undistort the image
    
#     undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)
    
#     # Define the aruco dictionary and charuco board
#     dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
#     board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
#     params = cv2.aruco.DetectorParameters()
#     #cv2.imwrite("test.png",undistorted_image)
#     # Detect markers in the undistorted image
#     marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(undistorted_image, dictionary, parameters=params)
    
#     # If at least one marker is detected
#     if len(marker_ids)>3:
#         # Interpolate CharUco corners
#         charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, undistorted_image, board)
        
#         # If enough corners are found, estimate the pose
#         if charuco_retval:
#             retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, None, None)

#             # If pose estimation is successful, draw the axis
#             if retval:
#                 cv2.drawFrameAxes(undistorted_image, camera_matrix, dist_coeffs, rvec, tvec, length=0.1, thickness=15)
#                 plt.imshow(undistorted_image)
#             else:
#                 return False,None,None
#         else:
#             return False,None,None
#         return charuco_retval,rvec,tvec
#     return False,None,None

def detect_pose(image, camera_matrix, dist_coeffs,board,debug_viz = False):
    # Use the original image for detection
    image_for_detection = image.copy()

    # Define the aruco dictionary and charuco board
    dictionary = board.getDictionary()
    # Customize detector parameters
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshConstant = 7  # Try increasing or decreasing to control thresholding
    params.adaptiveThreshWinSizeMin = 3  # Lowering window size can help with small markers
    params.adaptiveThreshWinSizeMax = 23  # Increase for more aggressive thresholding
    params.adaptiveThreshWinSizeStep = 10  # Size step in between windows
    
    params.minMarkerPerimeterRate = 0.04  # Smaller markers, so reduce the minimum perimeter
    params.maxMarkerPerimeterRate = 4.5   # Allow for larger perimeter in detection
    params.polygonalApproxAccuracyRate = 0.03  # Tighter approximation for more accurate corner detection
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # Use sub-pixel refinement for corners
    params.cornerRefinementMaxIterations = 50  # More iterations for better refinement
    params.cornerRefinementWinSize = 5  # Control the window size for corner refinement
    params.cornerRefinementMinAccuracy = 0.02  # Increase accuracy requirement for corner refinement
    
    # Detect markers in the original image
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
        image_for_detection, dictionary, parameters=params
    )
    print(len(marker_corners))
    # Proceed if markers are detected
    if marker_ids is not None:
        # Interpolate CharUco corners
        charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, image_for_detection, board
        )

        # Proceed if enough corners are found
        if charuco_retval and charuco_retval>=4:
            # Estimate the pose
            retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, None, None
            )
            if debug_viz:
                cv2.aruco.drawDetectedMarkers(image_for_detection, marker_corners, marker_ids)
                cv2.aruco.drawDetectedCornersCharuco(image_for_detection, charuco_corners, charuco_ids)
                cv2.drawFrameAxes(image_for_detection, camera_matrix, dist_coeffs, rvec, tvec, length=0.1)
                plt.imshow(cv2.cvtColor(image_for_detection, cv2.COLOR_BGR2RGB))
                plt.show()
            # Proceed if pose estimation is successful
            if retval:
                # Visualize detections and pose
                return True, rvec, tvec
            else:
                print("Not enough corners for pose estimation.")
                return False, None, None
        else:
            print("Not enough corners for pose estimation.")
            return False, None, None
    else:
        print("Not enough markers detected.")
        return False, None, None

