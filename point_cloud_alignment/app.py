import os
import numpy as np
import open3d as o3d
from flask import Flask, request, jsonify, send_file, render_template, after_this_request
import tempfile
import copy
import sys
sys.path.append(os.path.abspath(os.path.join("..",'lib')))

# Import the necessary modules
from calibration.cv2api.calibrate import read_chessboards, calibrate_camera
from calibration.cv2api.detect import detect_pose
from config.ConfigManagerServer import ConfigManagerAPI
from calibration.json_utils.json_functions import generate_json_for_images

print("Starting a calibration", flush=True)

config_manager = ConfigManagerAPI("http://127.0.0.1:5001")

# Get the absolute path of the directory where app.py is located
app_dir = os.path.abspath(os.path.dirname(__file__))

app = Flask(
    __name__,
    static_url_path='',
    static_folder=os.path.join(app_dir, 'static'),
    template_folder=os.path.join(app_dir, 'templates')
)

# Global variables to store point clouds
source_pcd = None
target_pcd = None
transformed_pcd = None


def align_point_clouds_with_icp(source, target, trans_init):
    """
    Perform ICP refinement with proper convergence criteria.
    """
    threshold = 10  # Adjust based on the data scale
    convergence_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=1000,  # Increase iterations for tighter convergence
        relative_fitness=1e-7,  # Lower for finer adjustments
        relative_rmse=1e-7  # Lower for better accuracy
    )
    
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, 
        target, 
        threshold, 
        trans_init, 
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False), 
        convergence_criteria
    )
    return reg_p2p.transformation

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_pointclouds', methods=['POST'])
def upload_pointclouds():
    global source_pcd, target_pcd
    source_file = request.files.get('source')
    target_file = request.files.get('target')

    try:
        if source_file:
            # Save source file to a temporary file
            with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as temp_source_file:
                source_file.save(temp_source_file)
                temp_source_filename = temp_source_file.name
            # Read point cloud from the temporary file
            source_pcd = o3d.io.read_point_cloud(temp_source_filename)
            # Delete the temporary file
            os.remove(temp_source_filename)

        if target_file:
            # Save target file to a temporary file
            with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as temp_target_file:
                target_file.save(temp_target_file)
                temp_target_filename = temp_target_file.name
            # Read point cloud from the temporary file
            target_pcd = o3d.io.read_point_cloud(temp_target_filename)
            # Delete the temporary file
            os.remove(temp_target_filename)

        return jsonify({'status': 'Point clouds uploaded successfully'})
    except Exception as e:
        print(f"Error uploading point clouds: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/align_pointclouds', methods=['POST'])
def align_pointclouds():
    global source_pcd, target_pcd, transformed_pcd

    data = request.get_json()
    source_points = np.array(data.get('source_points', []))
    target_points = np.array(data.get('target_points', []))

    print(f"Received {len(source_points)} source points")
    print(f"Received {len(target_points)} target points")

    if source_pcd is None or target_pcd is None:
        print("Point clouds not loaded")
        return jsonify({'error': 'Point clouds not loaded'}), 400

    if len(source_points) < 3 or len(target_points) < 3:
        print("Not enough points selected")
        return jsonify({'error': 'At least 3 points must be selected on each point cloud'}), 400
    
    if source_points.shape[0] != target_points.shape[0]:
            return jsonify({'error': 'The number of source points and target points must be the same.'}), 400

    try:
        # Create PointCloud objects from the selected points
        source_pc = o3d.geometry.PointCloud()
        source_pc.points = o3d.utility.Vector3dVector(source_points)

        target_pc = o3d.geometry.PointCloud()
        target_pc.points = o3d.utility.Vector3dVector(target_points)

        # Compute transformation using Procrustes (Kabsch algorithm)
        # Compute centroids
        source_centroid = np.mean(source_points, axis=0)
        target_centroid = np.mean(target_points, axis=0)

        # Center the points
        source_centered = source_points - source_centroid
        target_centered = target_points - target_centroid

        # Compute covariance matrix
        H = np.dot(source_centered.T, target_centered)

        # Singular Value Decomposition
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # Correct reflection if necessary
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)

        # Compute translation
        t = target_centroid - np.dot(R, source_centroid)

        # Construct transformation matrix
        trans_init = np.identity(4)
        trans_init[:3, :3] = R
        trans_init[:3, 3] = t

        # Apply the initial transformation to the source point cloud
        transformed_pcd = copy.deepcopy(source_pcd)
        #transformed_pcd.transform(trans_init)

        # Optionally, perform ICP refinement
        # Perform ICP refinement
        final_transformation = align_point_clouds_with_icp(transformed_pcd, target_pcd, trans_init)
        transformed_pcd.transform(final_transformation)

        # Convert transformed point cloud to list for JSON response
        transformed_points = np.asarray(transformed_pcd.points).tolist()

        return jsonify({'transformed_points': transformed_points})
    except Exception as e:
        print(f"Error in align_pointclouds: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/download_transformed_pcd')
def download_transformed_pcd():
    global transformed_pcd
    if transformed_pcd is None:
        return jsonify({'error': 'Transformed point cloud not available'}), 400

    try:
        # Save the transofmed point cloud on server
        
        current_work = config_manager.get_current_work()
        print(current_work)
        selected_project = current_work["selected_folder"]
        
        if(selected_project is not None):
            complete_path_node_red = os.path.join("../node-red/",selected_project, "aligned_pcd.ply")
            #
            o3d.io.write_point_cloud(complete_path_node_red, transformed_pcd)
            
            print(f"Saved under {complete_path_node_red}")
        else:
            print(f"Something went wrong while saving transformed pcd under selected_project")
        
        # Save the transformed point cloud to a temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.ply', delete=False)
        temp_filename = temp_file.name
        temp_file.close()  # Close the file so Open3D can write to it
        o3d.io.write_point_cloud(temp_filename, transformed_pcd)
        
        @after_this_request
        def remove_file(response):
            try:
                os.remove(temp_filename)
            except Exception as error:
                print(f"Error removing or closing downloaded file: {error}")
            return response

        return send_file(
            temp_filename,
            as_attachment=True,
            download_name='aligned_pcd.ply',  # Specify the filename here
            mimetype='application/octet-stream'
        )
    except Exception as e:
        print(f"Error during file writing or sending: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
