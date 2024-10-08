import sys
import threading
import os
from socket import socket
import argparse

import dash
from dash import dcc, html, ctx
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import open3d as o3d
import numpy as np
from dash.exceptions import PreventUpdate

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
                legend=dict(orientation='h')  # Ensure legend is horizontally oriented
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
    html.Div(id='container-ctx-example'),
    html.Div([
        html.Button('Run ICP Algorithm', id='icp-alg-button', n_clicks=0)
    ]),
])

# Store selected point indices
selected_point_indices = set()
selected_points = {obj_name: [] for obj_name in point_clouds.keys()}
count_n = 0
test = None

transformed_objects = {}

# Callback to capture clicks on points and update selected point style
@app.callback(
    Output('point-cloud-plot', 'figure', allow_duplicate=True),
    [Input('point-cloud-plot', 'clickData')],
    [State('point-cloud-plot', 'figure')],
    prevent_initial_call=True
)
def display_click_data(clickData, figure):
    global selected_point_indices, selected_points, test, count_n

    if clickData is not None:
        count_n += 1
        test = clickData
        point_clicked = clickData['points'][0]
        object_id = point_clicked['customdata']
        selected_point_index = point_clicked['pointNumber']
        obj_name = list(point_clouds.keys())[object_id - 1]

        # Check if the clicked point is already in the set of selected points
        if (object_id, selected_point_index) in selected_point_indices:
            selected_point_indices.remove((object_id, selected_point_index))
            selected_points[obj_name].remove([point_clicked["x"], point_clicked["y"], point_clicked["z"]])
        else:
            selected_point_indices.add((object_id, selected_point_index))
            selected_points[obj_name].append([point_clicked["x"], point_clicked["y"], point_clicked["z"]])

        new_figure = figure
        for i, trace in enumerate(new_figure['data']):
            trace_color = colors[i % len(colors)]  # Original color
            selected_color = 'green'
            # Update marker colors
            new_figure['data'][i]['marker']['color'] = [
                selected_color if (i + 1, j) in selected_point_indices else trace_color
                for j in range(len(trace['x']))
            ]
        return new_figure
    else:
        raise PreventUpdate

# Callback to show selected points
@app.callback(
    Output('show-points-output', 'children'),
    [Input('show-points-button', 'n_clicks')],
)
def show_points_button(n_clicks):
    if n_clicks > 0:
        return [html.P(f'Selected points for {obj_name}: {points}') for obj_name, points in selected_points.items()]
    else:
        raise PreventUpdate

# Callback to clear selected points
@app.callback(
    Output('clear-points-output', 'children'),
    [Input('clear-points-button', 'n_clicks')],
)
def clear_points_button(n_clicks):
    global selected_points
    if n_clicks > 0:
        selected_points = {obj_name: [] for obj_name in point_clouds.keys()}
        return "Points cleared."
    else:
        raise PreventUpdate

# Callback to calculate transformation matrix when button is clicked
@app.callback(
    Output('point-cloud-plot', 'figure', allow_duplicate=True),
    [Input('calculate-matrix-button', 'n_clicks')],
    [State('point-cloud-plot', 'figure')],
    prevent_initial_call=True

)
def calculate_transformation_matrix(n_clicks, figure):
    global selected_point_obj1
    global selected_point_obj2
    global transformed_object1
    if n_clicks is not None:

        src_points = np.array(selected_point_obj1)
        dst_points = np.array(selected_point_obj2)

        similarity_matrix = compute_similarity_transform(src_points, dst_points)

        transformed_all_points = apply_similarity_transform(np.asarray(points_object1), similarity_matrix)
        transformed_object1 = transformed_all_points
        np.save("transofrmed_frog1.npy", transformed_object1)
        figure["data"].append(go.Scatter3d(
            x=transformed_all_points[:, 0],
            y=transformed_all_points[:, 1],
            z=transformed_all_points[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color='black',
                opacity=0.8
            ),
            name='Transformed Points'
        ))
        return figure

    else:
        raise PreventUpdate


# Callback to calculate transformation matrix when button is clicked
@app.callback(
    Output('point-cloud-plot', 'figure'),
    [Input('icp-alg-button', 'n_clicks')],
    [State('point-cloud-plot', 'figure')],
    prevent_initial_call=True

)
def run_icp_alg(n_clicks, figure):
    global selected_point_obj1
    global selected_point_obj2
    global transformed_object1
    if n_clicks is not None:

        src_points = np.array(transformed_object1)
        dst_points = np.array(points_object2)

        obj_mov = PointCloud(src_points, columns=['x', 'y', 'z'])
        obj_fix = PointCloud(dst_points, columns=['x', 'y', 'z'])

        icp = SimpleICP()
        icp.add_point_clouds(obj_fix, obj_mov)
        H, X_mov_transformed, rigid_body_transformation_params, distance_residuals = icp.run(max_iterations=500)

        figure["data"].append(go.Scatter3d(
            x=X_mov_transformed[:, 0],
            y=X_mov_transformed[:, 1],
            z=X_mov_transformed[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color='brown',
                opacity=1
            ),
            name='ICPed Points'
        ))

        return figure

    else:
        raise PreventUpdate


# Function to stop the Dash app
def stop_dash():
    with open('exit_example.txt', 'w') as file:
        # Write a line to the file
        file.write("This is a line written to the text file.")
    app.server.stop()
    sys.exit()

# TCP/IP socket server thread
def socket_server():
    HOST = '127.0.0.1'  # localhost
    PORT = 65432        # Port to listen on

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                if data.decode() == 'stop':
                    print('Stop signal received from Node-RED')
                    stop_dash()

# Start TCP/IP socket server in a separate thread
threading.Thread(target=socket_server).start()


# TCP/IP socket server and Dash app execution code remains the same

if __name__ == '__main__':
    app.run_server(debug=True, port=6001)
