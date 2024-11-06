import open3d as o3d
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description='Filter outliers and extract largest connected component from a mesh.')
    parser.add_argument('input_mesh', type=str, help='Path to input mesh file.')
    parser.add_argument('output_mesh', type=str, help='Path to output mesh file.')
    parser.add_argument('--nb_neighbors', type=int, default=20, help='Number of neighbors to consider for outlier removal.')
    parser.add_argument('--std_ratio', type=float, default=2.0, help='Standard deviation ratio for outlier removal.')
    parser.add_argument('--do_poisson', action='store_true', help='Whether to perform Poisson reconstruction.')
    parser.add_argument('--voxel_size', type=float, default=None, help='Voxel size for downsampling.')
    parser.add_argument('--eps', type=float, default=0.02, help='DBSCAN eps parameter.')
    parser.add_argument('--min_points', type=int, default=10, help='DBSCAN min_points parameter.')
    parser.add_argument('--poisson_depth', type=int, default=15, help='Poisson reconstruction depth parameter.')
    parser.add_argument('--alpha', type=float, default=0.01, help='Alpha value for alpha shapes.')
    args = parser.parse_args()

    # Read the input mesh
    mesh = o3d.io.read_triangle_mesh(args.input_mesh)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    print("Input mesh has {} vertices and {} triangles.".format(len(mesh.vertices), len(mesh.triangles)),flush = True)

    # Convert mesh to point cloud
    pcd = mesh.sample_points_uniformly(number_of_points=len(mesh.vertices))
    print("Converted to point cloud with {} points.".format(len(pcd.points)),flush = True)

    # Optional downsampling
    if args.voxel_size is not None:
        pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)
        print("Downsampled point cloud to {} points.".format(len(pcd.points)),flush = True)

    # Remove outliers
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=args.nb_neighbors, std_ratio=args.std_ratio)
    pcd_filtered = pcd.select_by_index(ind)
    print("Point cloud after outlier removal has {} points.".format(len(pcd_filtered.points)),flush = True)

    # Find clusters using DBSCAN
    labels = np.array(pcd_filtered.cluster_dbscan(eps=args.eps, min_points=args.min_points, print_progress=True))
    max_label = labels.max()
    print("Point cloud has {} clusters.".format(max_label + 1),flush = True)

    if max_label < 0:
        print("No clusters found.",flush = True)
        return

    # Select the largest cluster
    counts = np.bincount(labels[labels >= 0])
    largest_cluster_idx = np.argmax(counts)
    indices = np.where(labels == largest_cluster_idx)[0]
    pcd_cluster = pcd_filtered.select_by_index(indices)
    print("Largest cluster has {} points.".format(len(pcd_cluster.points)),flush = True)

    # Ensure normals are estimated
    pcd_cluster.estimate_normals()
    pcd_cluster.orient_normals_consistent_tangent_plane(100)
    print("Normals estimated and oriented.",flush = True)

    # Optionally perform Poisson reconstruction
    if args.do_poisson:
        print("Performing Poisson reconstruction.")
        mesh_out, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd_cluster, depth=args.poisson_depth, linear_fit=True)
        print("Poisson reconstruction completed.")

        # Remove low-density vertices to eliminate spurious faces
        densities = np.asarray(densities)
        density_threshold = np.quantile(densities, 0.01)
        vertices_to_remove = densities < density_threshold
        mesh_out.remove_vertices_by_mask(vertices_to_remove)
        print("Removed low-density vertices.",flush = True)
        # Save the output mesh
        o3d.io.write_triangle_mesh(args.output_mesh, mesh_out)
        print("Saved output mesh to {}.".format(args.output_mesh),flush = True)
    else:
        # Save the output pcd
        o3d.io.write_point_cloud(args.output_mesh, pcd_cluster)
        print("Saved output point_cloud to {}.".format(args.output_mesh),flush = True)

    
    

if __name__ == "__main__":
    main()
