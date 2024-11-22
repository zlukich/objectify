import argparse
import trimesh
import numpy as np

def rescale_mesh(reference_mesh_path, target_mesh_path, output_path):
    # Load the meshes
    reference_mesh = trimesh.load(reference_mesh_path)
    target_mesh = trimesh.load(target_mesh_path)

    # Calculate the bounding box dimensions of both meshes
    reference_bbox = reference_mesh.bounding_box.extents
    target_bbox = target_mesh.bounding_box.extents

    # Calculate the scaling factor
    scale_factor = reference_bbox / target_bbox

    # Ensure uniform scaling by taking the average scaling factor
    uniform_scale_factor = np.mean(scale_factor)

    # Apply the scaling factor to the target mesh
    target_mesh.apply_scale(uniform_scale_factor)

    # Save the rescaled target mesh
    target_mesh.export(output_path)

    print(f"Target mesh rescaled and saved to: {output_path}")

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Rescale a target mesh to match the dimensions of a reference mesh.")
    parser.add_argument("reference_mesh", type=str, help="Path to the reference mesh file.")
    parser.add_argument("target_mesh", type=str, help="Path to the target mesh file.")
    parser.add_argument(
        "-o", "--output", type=str, default="rescaled_target_mesh.obj",
        help="Path to save the rescaled target mesh (default: rescaled_target_mesh.obj)."
    )

    args = parser.parse_args()

    # Call the rescale function with the provided arguments
    rescale_mesh(args.reference_mesh, args.target_mesh, args.output)

if __name__ == "__main__":
    main()
