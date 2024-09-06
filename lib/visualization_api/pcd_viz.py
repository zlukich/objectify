import sys
import threading
from socket import socket

import dash
from dash import dcc, html, ctx
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from dash.exceptions import PreventUpdate


import open3d as o3d

import numpy as np
from simpleicp import SimpleICP, PointCloud

### Functions
def compute_similarity_transform(src_points, dst_points):
    """
    Compute similarity transformation matrix based on corresponding points.

    Parameters:
        src_points (ndarray): Source point cloud, Nx3 array.
        dst_points (ndarray): Destination point cloud, Nx3 array.

    Returns:
        similarity_matrix (ndarray): 4x4 similarity transformation matrix.
    """
    # Compute centroids
    src_centroid = np.mean(src_points, axis=0)
    dst_centroid = np.mean(dst_points, axis=0)

    # Center the point clouds
    src_centered = src_points - src_centroid
    dst_centered = dst_points - dst_centroid

    # Compute scale
    scale = np.linalg.norm(dst_centered) / np.linalg.norm(src_centered)

    # Compute rotation using SVD
    H = np.dot(src_centered.T, dst_centered)
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # Construct similarity transformation matrix
    similarity_matrix = np.eye(4)
    similarity_matrix[:3, :3] = R * scale
    similarity_matrix[:3, 3] = dst_centroid - np.dot(similarity_matrix[:3, :3], src_centroid)

    return similarity_matrix


def apply_similarity_transform(points, similarity_matrix):
    """
    Apply similarity transformation to a set of points.

    Parameters:
        points (ndarray): Point cloud, Nx3 array.
        similarity_matrix (ndarray): 4x4 similarity transformation matrix.

    Returns:
        transformed_points (ndarray): Transformed point cloud, Nx3 array.
    """
    homogeneous_points = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed_points = np.dot(homogeneous_points, similarity_matrix.T)[:, :3]
    return transformed_points