import sys
import threading
import os
from socket import socket

import dash
from dash import dcc, html, ctx
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from dash.exceptions import PreventUpdate


import open3d as o3d

import numpy as np
from simpleicp import SimpleICP, PointCloud


sys.path.append(os.path.abspath(os.path.join("..",'lib')))

#from vizualization.viz_pcd import compute_similarity_transform,apply_similarity_transform 



frog1 = o3d.io.read_point_cloud("nerf_pc_high_res.ply").uniform_down_sample(every_k_points=40)
cl, ind = frog1.remove_statistical_outlier(nb_neighbors=20,
                                           std_ratio=2.0)
frog_points1 = cl.select_by_index(ind).points

frog2 = o3d.io.read_point_cloud("ngp_phone_pc_lowres.ply").uniform_down_sample(every_k_points=50)

cl, ind = frog2.remove_statistical_outlier(nb_neighbors=20,
                                           std_ratio=2.0)
frog_points2 = cl.select_by_index(ind).points

# Sample 3D point cloud data for two objects
points_object1 = frog_points1  # np.random.rand(10, 3)  # Sample points for object 1
points_object2 = frog_points2  # np.random.rand(10, 3)  # Sample points for object 2

# Add object identifier to each point
object1_id = np.ones((len(points_object1),), dtype=int)
object2_id = 2 * np.ones((len(points_object2),), dtype=int)

# Combine all points and their identifiers
all_points = np.concatenate((points_object1, points_object2))
all_ids = np.concatenate((object1_id, object2_id))

# Initialize Dash app
app = dash.Dash(__name__)

# Define initial layout with original points trace
initial_layout = go.Layout(
    scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z')
    )
)

# Define layout
app.layout = html.Div([
    dcc.Graph(
        id='point-cloud-plot', style={'width': '130vh', 'height': '85vh'},
        config={'scrollZoom': True},
        figure={
            'data': [
                go.Scatter3d(
                    x=all_points[all_ids == 1, 0],
                    y=all_points[all_ids == 1, 1],
                    z=all_points[all_ids == 1, 2],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color='blue',  # Change color for object 1
                        opacity=0.8
                    ),
                    name='Object 1',
                    customdata=all_ids[all_ids == 1],  # Associate object identifier with each point
                    #hoverinfo='skip',
                    #text=['Object 1: x={}, y={}, z={}'.format(x, y, z) for x, y, z in points_object1]
                    # Include XYZ coordinates in hover text
                ),
                go.Scatter3d(
                    x=all_points[all_ids == 2, 0],
                    y=all_points[all_ids == 2, 1],
                    z=all_points[all_ids == 2, 2],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color='red',  # Change color for object 2
                        opacity=0.8
                    ),
                    name='Object 2',
                    customdata=all_ids[all_ids == 2],  # Associate object identifier with each point
                    #hoverinfo='skip',
                    #text=['Object 2: x={}, y={}, z={}'.format(x, y, z) for x, y, z in points_object2]
                    # Include XYZ coordinates in hover text
                )
            ],
            'layout': go.Layout(
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

    html.Div(id='container-ctx-example'),

    html.Div([
        html.Button('Run ICP Algorithm', id='icp-alg-button', n_clicks=0)
    ]),
])

# Store selected point indices
selected_point_indices = set()

selected_point_obj1 = list()
selected_point_obj2 = list()
count_n = 0
test = None

transformed_object1 = None


# Callback to capture clicks on points and update selected point style
@app.callback(
    Output('point-cloud-plot', 'figure', allow_duplicate=True),
    [Input('point-cloud-plot', 'clickData')],
    [State('point-cloud-plot', 'figure')],
    prevent_initial_call=True
)
def display_click_data(clickData, figure):
    global selected_point_indices
    global selected_point_obj1
    global selected_point_obj2
    global test
    global count_n

    # if (count_n % 2 != 0):
    #     count_n = count_n + 1
    #     raise PreventUpdate

    if clickData is not None:
        count_n = count_n + 1
        test = clickData
        point_clicked = clickData['points'][0]
        object_id = point_clicked['customdata']
        selected_point_index = point_clicked['pointNumber']
        # Check if the clicked point is already in the set of selected points
        # Check if the clicked point is already selected
        if (object_id, selected_point_index) in selected_point_indices:
            # If already selected, remove it from the set of selected points
            selected_point_indices.remove((object_id, selected_point_index))

            # Update the figure to change the color back to original
            new_figure = figure
            for i, trace in enumerate(new_figure['data']):
                if i + 1 == object_id:
                    color = "blue" if object_id == 1 else "red"
                    new_figure['data'][i]['marker']['color'] = [
                        color if (i + 1, j) not in selected_point_indices else 'green' for j in range(len(trace['x']))]

            return new_figure  # Return the updated figure

        if (object_id == 1):
            selected_point_obj1.append([point_clicked["x"], point_clicked["y"], point_clicked["z"]])
        else:
            selected_point_obj2.append([point_clicked["x"], point_clicked["y"], point_clicked["z"]])

        if (object_id, selected_point_index) in selected_point_indices:
            selected_point_indices.remove((object_id, selected_point_index))
        else:
            selected_point_indices.add((object_id, selected_point_index))

        new_figure = figure

        for i, trace in enumerate(new_figure['data']):
            if i + 1 == object_id:  # Check if the trace belongs to the clicked object
                color = None
                if (object_id == 1):
                    color = "blue"
                else:
                    color = "red"
                new_figure['data'][i]['marker']['color'] = ['green' if (i + 1, j) in selected_point_indices else color
                                                            for j in range(len(trace['x']))]
        return new_figure
    else:
        raise PreventUpdate


# Callback to calculate transformation matrix when button is clicked
@app.callback(
    Output('show-points-output', 'children'),
    [Input('show-points-button', 'n_clicks')],
)
def show_points_button(n_clicks):
    if n_clicks > 0:
        # result = html.P(['Selected points object1:',' '.join(selected_point_obj1), html.Br(), 'Selected points object2:',' '.join(selected_point_obj2)])
        return (
        f'Selected points object1: {selected_point_obj1}', ' ', f'Selected points object2: {selected_point_obj2}')
    else:
        raise PreventUpdate


@app.callback(
    Output('clear-points-output', 'children'),
    [Input('clear-points-button', 'n_clicks')],
)
def clear_points_button(n_clicks):
    global selected_point_obj1
    global selected_point_obj2
    if n_clicks is not None:
        selected_point_obj1 = []
        selected_point_obj2 = []
        # show_points_button(n_clicks)
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

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True,port = 6001)




