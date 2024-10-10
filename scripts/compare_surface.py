import numpy as np
import open3d as o3d
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output, State
import matplotlib.colors as mcolors
from scipy.spatial import KDTree
import matplotlib

# Load point clouds
pcd1 = o3d.io.read_point_cloud("./data/meshes/reconstructed_frog.ply")
pcd2 = o3d.io.read_point_cloud("./data/meshes/ground_truth_frog.ply")

# Convert to numpy arrays
points1 = np.asarray(pcd1.points)
points2 = np.asarray(pcd2.points)

# Subsample the point clouds randomly to 50,000 points (or less if not enough points)
sample_size = 50000
indices1 = np.random.choice(len(points1), size=min(sample_size, len(points1)), replace=False)
indices2 = np.random.choice(len(points2), size=min(sample_size, len(points2)), replace=False)
subsampled_points1 = points1[indices1]
subsampled_points2 = points2[indices2]

# Build KDTree for the second point cloud
tree = KDTree(subsampled_points2)

# Compute distances from each point in subsampled_points1 to the nearest point in subsampled_points2
distances, _ = tree.query(subsampled_points1, k=1)

# Normalize distances for color mapping
min_distance = np.min(distances)
max_distance = np.max(distances)
norm = mcolors.Normalize(vmin=min_distance, vmax=max_distance)

# Map distances to colors using a colormap
cmap = matplotlib.colormaps['RdYlBu_r']  # Updated to fix deprecation warning
colors = [mcolors.rgb2hex(cmap(norm(d))) for d in distances]

# Initialize Dash app
app = Dash(__name__)

# Define initial figure
initial_figure = go.Figure(data=[
    go.Scatter3d(
        x=subsampled_points1[:, 0],
        y=subsampled_points1[:, 1],
        z=subsampled_points1[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=colors,
            colorscale='RdYlBu_r',
            showscale=True,
            colorbar=dict(title='Distance'),
        )
    )
])
initial_figure.update_layout(
    title='3D Scatter Plot',
    scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z')
    )
)

# Define app layout
app.layout = html.Div([
    dcc.Graph(
        id='3d-scatter-plot',
        style={'width': '100%', 'height': '90vh'},
        config={
            'scrollZoom': True,
            'displayModeBar': True,
            'displaylogo': False,
            'editable': False,
        },
        figure=initial_figure  # Set the initial figure here
    ),
    dcc.Slider(
        id='distance-slider',
        min=min_distance,
        max=max_distance,
        step=(max_distance - min_distance) / 100,
        value=max_distance,
        marks={round(min_distance + i * (max_distance - min_distance) / 5, 2): str(round(min_distance + i * (max_distance - min_distance) / 5, 2)) for i in range(6)},
    )
])

# # Update figure when slider value changes
@app.callback(
    Output('3d-scatter-plot', 'figure'),
    Input('distance-slider', 'value'),
    State('3d-scatter-plot', 'relayoutData')
)
def update_figure(selected_distance, relayout_data):
    # Filter points based on selected distance
    mask = distances <= selected_distance
    filtered_colors = [colors[i] if mask[i] else '#000000' for i in range(len(colors))]

    # Preserve camera position if available
    layout = go.Layout(
        title='3D Scatter Plot',
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        )
    )
    if relayout_data and 'scene.camera' in relayout_data:
        layout.scene.camera = relayout_data['scene.camera']

    # Update figure
    figure = go.Figure(data=[
        go.Scatter3d(
            x=subsampled_points1[:, 0],
            y=subsampled_points1[:, 1],
            z=subsampled_points1[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=filtered_colors,
                colorscale='RdYlBu_r',
                showscale=True,
                colorbar=dict(title='Distance'),
            )
        )
    ], layout=layout)

    return figure

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True, port=5454)
