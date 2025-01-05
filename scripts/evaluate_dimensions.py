import numpy as np
import trimesh
import json
import os

def calculate_bounding_box(mesh):
    """
    Calculate the bounding box dimensions of a mesh.

    Args:
        mesh (trimesh.Trimesh): The input mesh.

    Returns:
        dimensions (tuple): The (length, width, height) of the bounding box.
    """
    bounding_box = mesh.bounding_box
    bounds = bounding_box.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    dimensions = bounds[1] - bounds[0]  # [length, width, height]
    return dimensions

def compare_mesh_dimensions(reconstructed_path, ground_truth_path):
    """
    Compare the dimensions of a reconstructed mesh with the ground truth.

    Args:
        reconstructed_path (str): Path to the reconstructed mesh file.
        ground_truth_path (str): Path to the ground truth mesh file.

    Returns:
        results (dict): A dictionary with the dimensions and percentage errors.
    """
    # Load meshes
    reconstructed_mesh = trimesh.load(reconstructed_path)
    ground_truth_mesh = trimesh.load(ground_truth_path)

    # Calculate dimensions
    reconstructed_dimensions = calculate_bounding_box(reconstructed_mesh)
    ground_truth_dimensions = calculate_bounding_box(ground_truth_mesh)

    # Calculate percentage error for each dimension
    errors = ((reconstructed_dimensions - ground_truth_dimensions) / ground_truth_dimensions) * 100

    # Prepare results
    results = {
        "Reconstructed Path": reconstructed_path,
        "Ground Truth Path": ground_truth_path,
        "Reconstructed Dimensions": reconstructed_dimensions.tolist(),
        "Ground Truth Dimensions": ground_truth_dimensions.tolist(),
        "Percentage Errors": errors.tolist()
    }

    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare mesh dimensions.")
    parser.add_argument("reconstructed_mesh", help="Path to the reconstructed mesh file.")
    parser.add_argument("ground_truth_mesh", help="Path to the ground truth mesh file.")
    args = parser.parse_args()

    output_file = "dimensions_comparison_result.json"

    # Compare meshes
    results = compare_mesh_dimensions(args.reconstructed_mesh, args.ground_truth_mesh)

    # Save results to JSON file
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            all_results = json.load(f)
    else:
        all_results = []

    all_results.append(results)

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=4)

    print("Comparison results saved to", output_file)
