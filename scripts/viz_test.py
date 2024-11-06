import sys
import threading
import os
from socket import socket
import argparse

import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import open3d as o3d
import numpy as np
from dash.exceptions import PreventUpdate
import copy

from simpleicp import SimpleICP, PointCloud

sys.path.append(os.path.abspath(os.path.join("..",'lib')))

from visualization_api.pcd_viz import compute_similarity_transform,apply_similarity_transform

# Function to handle command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Visualize multiple point clouds.")
    parser.add_argument('files', nargs='+', help='Paths to the point cloud files.')
    return parser.parse_args()

# Load point clouds and process them
def load_point_clouds(file_paths):
    point_clouds = {}
    for i, file_path in enumerate(file_paths):
        obj_name = os.path.splitext(os.path.basename(file_path))[0]
        pcd = o3d.io.read_point_cloud(file_path).uniform_down_sample(every_k_points=40)
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        points = np.asarray(cl.select_by_index(ind).points)
        point_clouds[obj_name] = points
    return point_clouds

# Parse command-line arguments
args = parse_args()
file_paths = args.files
point_clouds = load_point_clouds(file_paths)

# Combine all points and assign unique IDs to each object
all_points = np.concatenate([points for points in point_clouds.values()])
all_ids = np.concatenate([np.full(len(points), i+1) for i, points in enumerate(point_clouds.values())])

# Initialize Dash app
app = dash.Dash(__name__)

# Create traces for each object
colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan']  # Extend as needed
traces = []
for i, (obj_name, points) in enumerate(point_clouds.items()):
    traces.append(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=6,
                color=colors[i % len(colors)],  # Alternate colors for objects
                opacity=0.8
            ),
            name=obj_name,
            customdata=np.full(len(points), i+1),
        )
    )

# Define layout
app.layout = html.Div([
    dcc.Graph(
        id='point-cloud-plot', style={'width': '130vh', 'height': '85vh'},
        config={'scrollZoom': True},
        figure={
            'data': traces,
            'layout': go.Layout(
                scene=dict(
                    xaxis=dict(title='X'),
                    yaxis=dict(title='Y'),
                    zaxis=dict(title='Z')
                ),
                margin=dict(l=0, r=0, b=0, t=0),
                legend=dict(orientation='h')
            )
        }
    ),
    html.Div([
        html.Button('Calculate Transformation Matrix', id='calculate-matrix-button', n_clicks=0),
        html.Div(id='transformation-matrix-output')
    ]),
    html.Div([
        html.Button('Check selected points', id='show-points-button', n_clicks=0),
        html.Div(id='show-points-output')
    ]),
    html.Div([
        html.Button('Clear selected points', id='clear-points-button', n_clicks=0),
        html.Div(id='clear-points-output')
    ]),
    html.Div([
        html.Button('Run ICP Algorithm', id='icp-alg-button', n_clicks=0)
    ]),
    # Add dcc.Store components
    dcc.Store(id='selected-point-indices-store', data=[]),
    dcc.Store(id='selected-points-store', data={obj_name: [] for obj_name in point_clouds.keys()}),
])

# Store transformed objects
transformed_objects = {}

# Callback to update selection data
@app.callback(
    Output('selected-point-indices-store', 'data', allow_duplicate=True),
    Output('selected-points-store', 'data', allow_duplicate=True),
    Input('point-cloud-plot', 'clickData'),
    State('selected-point-indices-store', 'data'),
    State('selected-points-store', 'data'),
    prevent_initial_call=True
)
def update_selection(clickData, selected_point_indices_data, selected_points_data):
    if clickData is None:
        raise PreventUpdate

    # Convert stored data back to appropriate types
    selected_point_indices = set(tuple(idx) for idx in selected_point_indices_data)
    selected_points = selected_points_data

    point_clicked = clickData['points'][0]
    object_id = point_clicked['customdata']
    selected_point_index = point_clicked['pointNumber']
    obj_name = list(point_clouds.keys())[object_id - 1]

    # Toggle selection
    if (object_id, selected_point_index) in selected_point_indices:
        selected_point_indices.remove((object_id, selected_point_index))
        selected_points[obj_name].remove([point_clicked["x"], point_clicked["y"], point_clicked["z"]])
    else:
        selected_point_indices.add((object_id, selected_point_index))
        selected_points[obj_name].append([point_clicked["x"], point_clicked["y"], point_clicked["z"]])

    # Prepare data for storage
    selected_point_indices_data = [list(idx) for idx in selected_point_indices]

    return selected_point_indices_data, selected_points

# Callback to update the figure based on selection
@app.callback(
    Output('point-cloud-plot', 'figure', allow_duplicate=True),
    Input('selected-point-indices-store', 'data'),
    State('point-cloud-plot', 'figure'),
    prevent_initial_call=True
)
def update_figure(selected_point_indices_data, figure):
    figure = copy.deepcopy(figure)

    selected_point_indices = set(tuple(idx) for idx in selected_point_indices_data)

    for i, trace in enumerate(figure['data']):
        trace_color = colors[i % len(colors)]  # Original color
        selected_color = 'green'
        # Update marker colors
        figure['data'][i]['marker']['color'] = [
            selected_color if (i + 1, j) in selected_point_indices else trace_color
            for j in range(len(trace['x']))
        ]

    return figure

# Callback to clear selected points
@app.callback(
    Output('selected-point-indices-store', 'data', allow_duplicate=True),
    Output('selected-points-store', 'data', allow_duplicate=True),
    Output('clear-points-output', 'children'),
    Input('clear-points-button', 'n_clicks'),
    prevent_initial_call=True
)
def clear_points(n_clicks):
    if n_clicks > 0:
        selected_point_indices_data = []
        selected_points_data = {obj_name: [] for obj_name in point_clouds.keys()}
        return selected_point_indices_data, selected_points_data, "Points cleared."
    else:
        raise PreventUpdate

# Callback to calculate transformation matrix
@app.callback(
    Output('point-cloud-plot', 'figure', allow_duplicate=True),
    Input('calculate-matrix-button', 'n_clicks'),
    State('point-cloud-plot', 'figure'),
    State('selected-points-store', 'data'),
    prevent_initial_call=True
)
def calculate_transformation_matrix(n_clicks, figure, selected_points):
    if n_clicks is None:
        raise PreventUpdate

    if all(len(points) >= 3 for points in selected_points.values()):
        obj_names = list(selected_points.keys())
        src_obj_name = obj_names[0]
        dst_obj_name = obj_names[1]

        src_points = np.array(selected_points[src_obj_name])
        dst_points = np.array(selected_points[dst_obj_name])

        similarity_matrix = compute_similarity_transform(src_points, dst_points)
        src_points_full = point_clouds[src_obj_name]
        transformed_points = apply_similarity_transform(src_points_full, similarity_matrix)

        figure = copy.deepcopy(figure)
        figure['data'].append(go.Scatter3d(
            x=transformed_points[:, 0],
            y=transformed_points[:, 1],
            z=transformed_points[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color='black',
                opacity=0.8
            ),
            name='Transformed Points'
        ))

        transformed_objects[src_obj_name] = transformed_points

        return figure
    else:
        print("Please select at least 3 points from each object.")
        raise PreventUpdate

# Callback to run ICP algorithm
@app.callback(
    Output('point-cloud-plot', 'figure', allow_duplicate=True),
    Input('icp-alg-button', 'n_clicks'),
    State('point-cloud-plot', 'figure'),
    prevent_initial_call=True
)
def run_icp_alg(n_clicks, figure):
    if n_clicks is None:
        raise PreventUpdate

    if transformed_objects:
        src_obj_name = list(transformed_objects.keys())[0]
        dst_obj_name = [name for name in point_clouds.keys() if name != src_obj_name][0]

        src_points = transformed_objects[src_obj_name]
        dst_points = point_clouds[dst_obj_name]

        obj_mov = PointCloud(src_points, columns=['x', 'y', 'z'])
        obj_fix = PointCloud(dst_points, columns=['x', 'y', 'z'])

        icp = SimpleICP()
        icp.add_point_clouds(obj_fix, obj_mov)
        H, X_mov_transformed, rigid_body_transformation_params, distance_residuals = icp.run(max_iterations=500)

        figure = copy.deepcopy(figure)
        figure['data'].append(go.Scatter3d(
            x=X_mov_transformed[:, 0],
            y=X_mov_transformed[:, 1],
            z=X_mov_transformed[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color='brown',
                opacity=1
            ),
            name='ICP Transformed Points'
        ))

        return figure
    else:
        print("Please calculate the transformation matrix first.")
        raise PreventUpdate

# Callback to show selected points
@app.callback(
    Output('show-points-output', 'children'),
    Input('show-points-button', 'n_clicks'),
    State('selected-points-store', 'data'),
    prevent_initial_call=True
)
def show_points_button(n_clicks, selected_points_data):
    if n_clicks > 0:
        return [html.P(f'Selected points for {obj_name}: {points}') for obj_name, points in selected_points_data.items()]
    else:
        raise PreventUpdate

if __name__ == '__main__':
    app.run_server(debug=True, port=6001)
