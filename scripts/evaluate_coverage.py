import os
import numpy as np
import trimesh
import open3d as o3d
from scipy.spatial import cKDTree
import json

def symmetric_chamfer_distance(points1, points2):
    """
    Compute the symmetric Chamfer distance between two point sets.
    """
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)
    distances1, _ = tree1.query(points2, k=1)  # GT -> Recon
    distances2, _ = tree2.query(points1, k=1)  # Recon -> GT
    chamfer_dist = (np.mean(distances1**2) + np.mean(distances2**2)) / 2
    return chamfer_dist

def symmetric_hausdorff_distance(points1, points2):
    """
    Compute the symmetric Hausdorff distance between two point sets.
    """
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)
    distances1, _ = tree1.query(points2, k=1)  # GT -> Recon
    distances2, _ = tree2.query(points1, k=1)  # Recon -> GT
    hausdorff_dist = max(np.max(distances1), np.max(distances2))
    return hausdorff_dist

def point_cloud_coverage(gt_points, recon_points, threshold=0.01):
    """
    Calculate the coverage of reconstructed points relative to ground truth points.

    Args:
        gt_points (np.ndarray): Ground truth point cloud.
        recon_points (np.ndarray): Reconstructed point cloud.
        threshold (float): Distance threshold for coverage.

    Returns:
        float: Coverage ratio.
    """
    tree = cKDTree(recon_points)
    distances, _ = tree.query(gt_points, k=1)
    coverage = np.sum(distances < threshold) / len(gt_points)
    return coverage

def f_score(gt_points, recon_points, threshold=0.01):
    """
    Calculate the F-Score between two point clouds.

    Args:
        gt_points (np.ndarray): Ground truth point cloud.
        recon_points (np.ndarray): Reconstructed point cloud.
        threshold (float): Distance threshold for precision and recall.

    Returns:
        float: F-Score.
    """
    tree_gt = cKDTree(gt_points)
    tree_recon = cKDTree(recon_points)

    distances_gt_to_recon, _ = tree_recon.query(gt_points, k=1)
    distances_recon_to_gt, _ = tree_gt.query(recon_points, k=1)

    precision = np.sum(distances_recon_to_gt < threshold) / len(recon_points)
    recall = np.sum(distances_gt_to_recon < threshold) / len(gt_points)

    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def downsample_point_cloud(pcd, target_points):
    """
    Downsample a point cloud to a target number of points.

    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        target_points (int): The desired number of points.

    Returns:
        np.ndarray: Downsampled points.
    """
    points = np.asarray(pcd.points)
    if len(points) > target_points:
        indices = np.random.choice(len(points), target_points, replace=False)
        return points[indices]
    return points

def update_json_with_colmap(json_file):
    """
    Update the JSON file to include Colmap point clouds.

    Args:
        json_file (str): Path to the JSON file containing point cloud paths.

    Returns:
        updated_data (list): Updated JSON data with Colmap paths.
    """
    with open(json_file, "r") as f:
        data = json.load(f)

    for entry in data:
        gt_dir = os.path.dirname(entry["Ground Truth Path"])
        colmap_path = os.path.join(gt_dir, "fused.ply")
        entry["Colmap Path"] = colmap_path

    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)

    print("JSON updated with Colmap paths.")
    return data

def compare_point_clouds(json_file, target_points=100000, threshold=0.02):
    """
    Compare ground truth, reconstructed, and Colmap point clouds using multiple metrics.

    Args:
        json_file (str): Path to the JSON file containing point cloud paths.
        target_points (int): Number of points to downsample each point cloud to.
        threshold (float): Threshold for ICP registration.

    Returns:
        table_data (dict): Data for table generation.
    """
    with open(json_file, "r") as f:
        data = json.load(f)

    table_data = {
        "reconstructed_vs_gt": [],
        "colmap_vs_gt": []
    }

    for entry in data:
        gt_path = entry["Ground Truth Path"]
        recon_path = entry["Reconstructed Path"]
        colmap_path = entry.get("Colmap Path")

        object_name = os.path.basename(os.path.dirname(gt_path))

        # Load point clouds
        gt_pcd = o3d.io.read_point_cloud(gt_path)
        recon_pcd = o3d.io.read_point_cloud(recon_path)
        if colmap_path and os.path.exists(colmap_path):
            colmap_pcd = o3d.io.read_point_cloud(colmap_path)
        else:
            print(f"Colmap point cloud not found for: {colmap_path}")
            colmap_pcd = None

        # Downsample point clouds
        gt_points = downsample_point_cloud(gt_pcd, target_points)
        recon_points = downsample_point_cloud(recon_pcd, target_points)
        if colmap_pcd:
            colmap_points = downsample_point_cloud(colmap_pcd, target_points)

        # Compute metrics for reconstructed vs GT
        chamfer_recon = symmetric_chamfer_distance(gt_points, recon_points)
        hausdorff_recon = symmetric_hausdorff_distance(gt_points, recon_points)
        coverage_recon = point_cloud_coverage(gt_points, recon_points, threshold)
        fscore_recon = f_score(gt_points, recon_points, threshold)

        table_data["reconstructed_vs_gt"].append({
            "Object": object_name,
            "Symmetric Chamfer Distance": chamfer_recon,
            "Symmetric Hausdorff Distance": hausdorff_recon,
            "Coverage": coverage_recon,
            "F-Score": fscore_recon
        })
    
        # Compute metrics for Colmap vs GT
        if colmap_pcd:
            chamfer_colmap = symmetric_chamfer_distance(gt_points, colmap_points)
            hausdorff_colmap = symmetric_hausdorff_distance(gt_points, colmap_points)
            coverage_colmap = point_cloud_coverage(gt_points, colmap_points, threshold) 
            fscore_colmap = f_score(gt_points, colmap_points, threshold)

            table_data["colmap_vs_gt"].append({
                "Object": object_name,
                "Symmetric Chamfer Distance": chamfer_colmap,
                "Symmetric Hausdorff Distance": hausdorff_colmap,
                "Coverage": coverage_colmap,
                "F-Score": fscore_colmap
            })

    # Save table data to JSON file
    output_file = "comparison_table.json"
    with open(output_file, "w") as f:
        json.dump(table_data, f, indent=4)

    print(f"Table data saved to {output_file}")
    return table_data

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare point clouds and generate table data.")
    parser.add_argument("json_file", help="Path to the JSON file containing point cloud paths.")
    parser.add_argument("--target_points", type=int, default=100000, help="Number of points to downsample each point cloud to.")
    parser.add_argument("--threshold", type=float, default=0.02, help="Distance threshold for metrics.")
    args = parser.parse_args()

    # Update JSON with Colmap paths
    #updated_data = update_json_with_colmap(args.json_file)

    # Perform comparisons and generate table data
    table_data = compare_point_clouds(args.json_file, args.target_points, args.threshold)

    print("Table Data:", table_data)
