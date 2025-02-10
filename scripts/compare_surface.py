
import numpy as np
import open3d as o3d
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output, State, no_update
import matplotlib.colors as mcolors
from scipy.spatial import KDTree
from scipy.spatial import cKDTree
import matplotlib
import argparse
from time import sleep

def chamfer_distance(tree1, tree2):
    # Create KD trees for efficient nearest-neighbor search

    # For each point in points1, find the closest point in points2
    distances1, _ = tree1.query(points2, k=1)
    distances2, _ = tree2.query(points1, k=1)

    # Compute the Chamfer distance
    chamfer_dist = np.mean(distances1**2) + np.mean(distances2**2)
    return chamfer_dist

# Function to calculate Hausdorff Distance
def hausdorff_distance(tree1, tree2):
    distances1, _ = tree1.query(points2, k=1)
    distances2, _ = tree2.query(points1, k=1)
    hausdorff_dist = max(np.max(distances1), np.max(distances2))
    return hausdorff_dist

parser = argparse.ArgumentParser(description="Comparing surfaces script")
parser.add_argument('--pcd_source', type=str, required=True, help='Path to source point cloud')
parser.add_argument('--pcd_target', type=str, required=True, help='Path to target point cloud')
args = parser.parse_args()

# Load point clouds
try:
    pcd1 = o3d.io.read_point_cloud(args.pcd_source)
    pcd2 = o3d.io.read_point_cloud(args.pcd_target)
    
    # Convert point clouds to numpy arrays for further processing
    points1 = np.asarray(pcd1.points)
    points2 = np.asarray(pcd2.points)   
    
    # Subsample the point clouds randomly
    sample_size = 10000
    indices1 = np.random.choice(len(points1), size=min(sample_size, len(points1)), replace=False)
    indices2 = np.random.choice(len(points2), size=min(sample_size, len(points2)), replace=False)
    subsampled_points1 = points1[indices1]
    subsampled_points2 = points2[indices2]

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(subsampled_points1)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(subsampled_points2)
    print("Number of points" , len(pcd1.points),flush = True)
except:
    raise Exception("Error while reading files occurred")
print('Reading of PCDs ended',flush = True)  
# Perform ICP alignment
threshold = 0.02  # Distance threshold for ICP
transformation_init = np.identity(4)  # Initial alignment guess
sleep(15)
# Run ICP
reg_p2p = o3d.pipelines.registration.registration_icp(
    pcd1, pcd2, threshold, transformation_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling = False),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
)

icp_fitness = reg_p2p.fitness
icp_inlier_rmse = reg_p2p.inlier_rmse


# Apply the transformation to align point clouds
pcd1.transform(reg_p2p.transformation)

print('ICP alignment finisher',flush = True)

tree1 = cKDTree(subsampled_points1)
tree2 = cKDTree(subsampled_points2)
# Calculate Chamfer Distance and Hausdorff Distance after ICP alignment
chamfer_dist = chamfer_distance(tree1, tree2)
hausdorff_dist = hausdorff_distance(tree1, tree2)

# Build KDTree for the second point cloud
tree = cKDTree(subsampled_points2)

# Compute distances
distances, _ = tree.query(subsampled_points1, k=1)
print(' Building of KDTREE finished' ,flush = True)  
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

# Initialize Dash app
app = Dash(__name__)

# Define initial 3D scatter plot figure
scatter_fig = go.Figure(data=[
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
scatter_fig.update_layout(
    title='3D Scatter Plot',
    scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z')
    )
)

# Custom histogram creation with color distribution
num_bins = 30
hist, bin_edges = np.histogram(distances, bins=num_bins)
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
bin_colors = [mcolors.rgb2hex(cmap(norm(bc))) for bc in bin_centers]

histogram_data = [
    go.Bar(
        x=bin_centers,
        y=hist,
        marker=dict(color=bin_colors),
        width=(max_distance - min_distance) / num_bins
    )
]

# Create histogram figure
hist_fig = go.Figure(data=histogram_data)
hist_fig.update_layout(
    title='Distance Distribution',
    xaxis=dict(title='Distance'),
    yaxis=dict(title='Frequency'),
    bargap=0.05,
)

# Define app layout with both plots
app.layout = html.Div([
    html.Div([
        html.P(f"Chamfer Distance: {chamfer_dist:.10f}"),
        html.P(f"Hausdorff Distance: {hausdorff_dist:.10f}"),
        html.P(f"ICP Fitness: {icp_fitness:.4f}"),
        html.P(f"ICP Inlier RMSE: {icp_inlier_rmse:.4f}")
    ], style={
        'fontSize': '14px', 
        'textAlign': 'center', 
        'padding': '5', 
        'background': '#eee', 
        'borderRadius': '3px',
        'marginBottom': '5px',
        'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)',
        'lineHeight': '1.6'
    }),
    
    
    # 3D Scatter Plot
    dcc.Graph(
        id='3d-scatter-plot',
        style={'width': '100%', 'height': '70vh'},
        config={
            'scrollZoom': True,
            'displayModeBar': True,
            'displaylogo': False,
            'editable': False,
        },
        figure=scatter_fig
    ),
    
    # Container for both the histogram and the slider with fixed width
    html.Div([
        # Distance Distribution Histogram
        dcc.Graph(
            id='distance-histogram',
            style={'width': '100%', 'height': '10vh', 'margin': '0'},  # Make histogram use full width
            figure=hist_fig.update_layout(
                xaxis=dict(range=[min_distance, max_distance], title='Distance', fixedrange=True),
                yaxis={'visible': False, 'showticklabels': False},
                margin=dict(l=0, r=0, t=40, b=40),  # Adjust margins to match slider width
                bargap=0.05,  # Adjust the spacing between bars if needed
            )
        ),
        
        
        html.Div(f"Filter points based on their distance to neighbours", style={'fontSize': 18, 'textAlign': 'left', 'marginBottom': '10px', 'background':'white'}),
        # Distance Slider with a slightly wider width wrapped in an html.Div
        html.Div(
            
            dcc.Slider(
                id='distance-slider',
                min=round(float(min_distance), 2),
                max=round(float(max_distance), 2),
                step=round((max_distance - min_distance) / num_bins, 2) if max_distance > min_distance else 1,
                value=float(max_distance),
                marks={str(round(float(i), 2)): str(round(float(i), 2)) 
                       for idx, i in enumerate(np.linspace(min_distance, max_distance, num_bins)) if idx % 5 == 0},  # Show every 5th mark
                included=True,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            style={'width': '101%', 'margin-left': '-0.5%'}  # Apply style here instead of directly on the slider
        ),
    ], style={'width': '100%', 'margin': '0 auto'}),  # Set the container to 80% of the page width, centered
])


# Update scatter plot when slider value changes
@app.callback(
    Output('3d-scatter-plot', 'figure'),
    Input('distance-slider', 'value')
)
def update_scatter(selected_distance):
    # Filter points based on selected distance
    mask = distances <= selected_distance
    filtered_colors = ['#FF0000' if not mask[i] else colors[i] for i in range(len(colors))]

    # Update figure
    scatter_fig = go.Figure(data=[
        go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=3,
                color=filtered_colors,
                colorscale='RdYlBu_r',
                showscale=True,
                colorbar=dict(title='Distance'),
            )
        )
    ])
    scatter_fig.update_layout(
        title='3D Scatter Plot',
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        )
    )
    return scatter_fig

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=False, port=5454)
