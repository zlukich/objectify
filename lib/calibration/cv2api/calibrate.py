import cv2
import numpy as np

# ARUCO_DICT = cv2.aruco.DICT_4X4_50
# SQUARES_VERTICALLY = 7
# SQUARES_HORIZONTALLY = 5
# SQUARE_LENGTH = 0.056
# MARKER_LENGTH = 0.042

ARUCO_DICT = cv2.aruco.DICT_ARUCO_ORIGINAL
SQUARES_VERTICALLY = 10
SQUARES_HORIZONTALLY = 7
SQUARE_LENGTH = 0.03
MARKER_LENGTH = 0.015


# ...
#PATH_TO_YOUR_IMAGES = './images/solenoid/' # example
# ------------------------------
dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)

def read_chessboards(images):
    """
    Charuco base pose estimation.
    """
    print("POSE ESTIMATION STARTS:")
    
    allCorners = []
    allIds = []

    all_corners = np.array([])
    all_ids = np.array([])
    num_detected_markers = []
    decimator = 0
    # SUB PIXEL CORNER DETECTION CRITERION
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
    imsize = -1
    for im in images:
        print("=> Processing image {0}".format(im))
        frame = cv2.imread(im)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        imsize = gray.shape
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, dictionary)

        if len(corners)>0:
            # SUB PIXEL DETECTION
            for corner in corners:
                cv2.cornerSubPix(gray, corner,
                                 winSize = (3,3),
                                 zeroZone = (-1,-1),
                                 criteria = criteria)
            res2 = cv2.aruco.interpolateCornersCharuco(corners,ids,gray,board)
            if res2[1] is not None and res2[2] is not None and len(res2[1])>6 and decimator%1==0:
                allCorners.append(res2[1])
                allIds.append(res2[2])

            decimator+=1

    
            # marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
            # charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray, board)
            # if charuco_retval and len(charuco_corners>3):
            #     allCorners.append(charuco_corners)
            #     allIds.append(charuco_ids)

    #         if np.size(all_corne|rs) == 0:
    #             all_corners = corners
    #             all_ids = ids
    #         else:
    #             all_corners = np.append(all_corners, corners, axis=0)
    #             all_ids = np.append(all_ids, ids, axis=0)
    #         num_detected_markers.append(len(ids))

            

    # num_detected_markers = np.array(num_detected_markers)

    # allCorners = all_corners
    # allIds= all_ids

    
    return allCorners,allIds,imsize,num_detected_markers

def calibrate_camera(allCorners,allIds,imsize):
    """
    Calibrates the camera using the dected corners.
    """
    print("CAMERA CALIBRATION")

    # cameraMatrixInit = np.array([[ 1000.,    0., imsize[0]/2.],
    #                              [    0., 1000., imsize[1]/2.],
    #                              [    0.,    0.,           1.]])

    # distCoeffsInit = np.zeros((5,1))
    # flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
    # #flags = (cv2.CALIB_RATIONAL_MODEL)
    # retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(allCorners, allIds, board, imsize, None, None)

    # return retval,camera_matrix,dist_coeffs,rvecs,tvecs
    cameraMatrixInit = np.array([[ 1000.,    0., imsize[0]/2.],
                                 [    0., 1000., imsize[1]/2.],
                                 [    0.,    0.,           1.]])

    distCoeffsInit = np.zeros((5,1))
    #flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
    flags = (cv2.CALIB_RATIONAL_MODEL)
    (ret, camera_matrix, distortion_coefficients0,
     rotation_vectors, translation_vectors,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics,
     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
                      charucoCorners=allCorners,
                      charucoIds=allIds,
                      board=board,
                      imageSize=imsize,
                      cameraMatrix=cameraMatrixInit,
                      distCoeffs=distCoeffsInit,
                      flags=flags,
                      criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors