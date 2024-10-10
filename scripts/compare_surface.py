import numpy as np
import open3d as o3d
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output, State, no_update
import matplotlib.colors as mcolors
from scipy.spatial import KDTree
import matplotlib
import math
import traceback
import logging
#logging.basicConfig(level=logging.DEBUG)


# Load point clouds
pcd1 = o3d.io.read_point_cloud("./data/meshes/reconstructed_frog.ply")
pcd2 = o3d.io.read_point_cloud("./data/meshes/ground_truth_frog.ply")

# Convert to numpy arrays
points1 = np.asarray(pcd1.points)
points2 = np.asarray(pcd2.points)

# Subsample the point clouds randomly
sample_size = 50000
indices1 = np.random.choice(len(points1), size=min(sample_size, len(points1)), replace=False)
indices2 = np.random.choice(len(points2), size=min(sample_size, len(points2)), replace=False)
subsampled_points1 = points1[indices1]
subsampled_points2 = points2[indices2]

# Build KDTree for the second point cloud
tree = KDTree(subsampled_points2)

# Compute distances
distances, _ = tree.query(subsampled_points1, k=1)

# Remove NaN and infinite distances
valid_mask = np.isfinite(distances)
distances = distances[valid_mask]
subsampled_points1 = subsampled_points1[valid_mask]

# Normalize distances for color mapping
min_distance = np.min(distances)
max_distance = np.max(distances)
norm = mcolors.Normalize(vmin=min_distance, vmax=max_distance)

# Map distances to colors using a colormap
cmap = matplotlib.colormaps['RdYlBu_r']
colors = [mcolors.rgb2hex(cmap(norm(d))) for d in distances]

# Convert arrays to lists for JSON serialization
x_coords = subsampled_points1[:, 0].tolist()
y_coords = subsampled_points1[:, 1].tolist()
z_coords = subsampled_points1[:, 2].tolist()

print(f"Number of points: {len(x_coords)}")
print(f"Number of distances: {len(distances)}")
print(f"Min distance: {min_distance}, Max distance: {max_distance}")

if not np.all(np.isfinite(distances)):
    print("Warning: Non-finite values found in distances.")
    
print(f"Lengths - X: {len(x_coords)}, Y: {len(y_coords)}, Z: {len(z_coords)}, Colors: {len(colors)}")


# Initialize Dash app
app = Dash(__name__)

# Define initial figure
initial_figure = go.Figure(data=[
    go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
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
        figure=initial_figure
    ),
    dcc.Slider(
        id='distance-slider',
        min=round(float(min_distance),2),
        max=round(float(max_distance),2),
        step=round((max_distance - min_distance) / 10,2) if max_distance > min_distance else 1,
        value=float(max_distance),
        
    )
])

# Update figure when slider value changes
@app.callback(
    Output('3d-scatter-plot', 'figure'),
    Input('distance-slider', 'value')
)
def update_figure(selected_distance):
    print(f"Selected distance: {selected_distance}")
    # Filter points based on selected distance
    mask = distances <= selected_distance
    filtered_colors = ['#FF0000' if not mask[i] else colors[i] for i in range(len(colors))]  # Make filtered points red
    
    # Increase the size of filtered points
    filtered_sizes = [3 if not mask[i] else 2 for i in range(len(colors))]  # Larger size for filtered points

    # Update data with lists for JSON serialization
    updated_x = x_coords
    updated_y = y_coords
    updated_z = z_coords

    # Preserve camera position if available
    layout = go.Layout(
        title='3D Scatter Plot',
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        )
    )

    # Update figure
    figure = go.Figure(data=[
        go.Scatter3d(
            x=updated_x,
            y=updated_y,
            z=updated_z,
            mode='markers',
            marker=dict(
                size=3,
                color=filtered_colors,
                #size = filtered_sizes,
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
