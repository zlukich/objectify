import json
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio




def get_frustum_vertices(scale=0.1):
    """Define frustum vertices in camera coordinates."""
    return np.array([
        [0, 0, 0],  # Camera center
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1]
    ]) * scale

def transform_frustum(vertices, transform_matrix):
    """Transform frustum vertices to world coordinates."""
    rotation_matrix = transform_matrix[:3, :3]
    camera_position = transform_matrix[:3, 3]
    vertices_world = (rotation_matrix @ vertices.T).T + camera_position
    return vertices_world

def camera_with_frustums(json_path):

    # Load JSON file containing transformation matrices
    with open(json_path, 'r') as f:
        data = json.load(f)


    # Extract frames from JSON data
    frames = data['frames']

    # Plot camera frustums
    fig = go.Figure()

    for frame in frames:
        transform_matrix = np.array(frame['transform_matrix'])
        frustum_vertices = get_frustum_vertices()
        frustum_vertices_world = transform_frustum(frustum_vertices, transform_matrix)
        
        # Plot camera center
        fig.add_trace(go.Scatter3d(
            x=[frustum_vertices_world[0, 0]], 
            y=[frustum_vertices_world[0, 1]], 
            z=[frustum_vertices_world[0, 2]],
            mode='markers',
            marker=dict(size=3, color='red'),
            name='Camera Center'
        ))
        
        # Plot frustum edges
        edges = [
            (0, 1), (0, 2), (0, 3), (0, 4),  # From camera center to frustum vertices
            (1, 2), (2, 3), (3, 4), (4, 1)   # Between frustum vertices
        ]
        
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=[frustum_vertices_world[edge[0], 0], frustum_vertices_world[edge[1], 0]],
                y=[frustum_vertices_world[edge[0], 1], frustum_vertices_world[edge[1], 1]],
                z=[frustum_vertices_world[edge[0], 2], frustum_vertices_world[edge[1], 2]],
                mode='lines',
                line=dict(color='red'),
                name='Frustum Edge'
            ))

    # Set plot labels
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title='Camera Frustums'
    )

    # Show plot
    pio.show(fig)