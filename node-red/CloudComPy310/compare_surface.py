import sys
import os
# Define the environment variables
script_dir = os.path.dirname(os.path.abspath(__file__))
cloudcompy_root = script_dir
sys.path.append('C:\\Users\\soboliev\\Desktop\\objectify\\node-red\\CloudComPy310\\CloudCompare')

# Modify environment variables for the current process
os.environ['SCRIPT_DIR'] = script_dir
os.environ['CLOUDCOMPY_ROOT'] = cloudcompy_root
os.environ['PYTHONPATH'] = f"{cloudcompy_root}\\CloudCompare;{os.environ.get('PYTHONPATH', '')}"
os.environ['PYTHONPATH'] += f";{cloudcompy_root}\\doc\\PythonAPI_test"
os.environ['PATH'] = f"{cloudcompy_root}\\CloudCompare;{cloudcompy_root}\\ccViewer;{script_dir};{os.environ.get('PATH', '')}"
os.environ['PATH'] += f";{cloudcompy_root}\\CloudCompare\\plugins"

# Print environment variables to verify changes
# print("SCRIPT_DIR:", os.environ['SCRIPT_DIR'])
# print("CLOUDCOMPY_ROOT:", os.environ['CLOUDCOMPY_ROOT'])
# print("PYTHONPATH:", os.environ['PYTHONPATH'])
# print("PATH:", os.environ['PATH'])

# Now you can run your script or subprocesses with the modified environment
# Example of running a Python script
# os.system(f"python {cloudcompy_root}\\compare_surface.py")


import cloudComPy as cc
import dash
from dash import dcc, html, Output, Input, State
import plotly.graph_objs as go
import numpy as np
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.cm
import traceback
#from PyQt5.QtCore import Q  # Import QString from PyQt5


if cc.isPluginM3C2():
    import cloudComPy.M3C2
    print("test")

try:
    #later pass distTest1 and distTest2 as parameters
    
    distTest1 = cc.loadPointCloud("../data/registered_compare_entity.ply")
    
    distTest2 = cc.loadPointCloud("../data/ground_truth_object.ply")
    
    refTest1 = cc.CloudSamplingTools.subsampleCloudRandomly(distTest1, 50000)
    print("something")
    (distTest1, res) = distTest1.partialClone(refTest1)
    refTest2 = cc.CloudSamplingTools.subsampleCloudRandomly(distTest2, 50000)
    (distTest2, res) = distTest2.partialClone(refTest2)
    distTest1.deleteAllScalarFields()
    distTest2.deleteAllScalarFields()
    
    
except Exception as e:
    print(f"An error occurred: {e}")
    traceback.print_exc()  # Prints the full traceback to help identify the issue
    raise  # re-raise the exception to exit with an error code





bestOctreeLevel = cc.DistanceComputationTools.determineBestOctreeLevel(distTest1, None, distTest2)
params = cc.Cloud2CloudDistancesComputationParams()
params.maxThreadCount = 12
params.octreeLevel = bestOctreeLevel
stats = cc.DistanceComputationTools.computeCloud2CloudDistances(distTest1, distTest2, params)

# params = cc.M3C2.M3C2guessParamsToFile([distTest1,distTest2], "test.txt",True)
# print("parameters written")

# output_cloud = cc.M3C2.computeM3C2([distTest1,distTest2],"test.txt")
# print("M3C2 Distance computed")

dic = distTest1.getScalarFieldDic()
print(dic)
sf = distTest1.getScalarField(dic['C2C absolute distances'])
distances = sf.toNpArray()
distTest1.setCurrentDisplayedScalarField(1)
distTest1.convertCurrentScalarFieldToColors()
cola = distTest1.colorsToNpArray()
cola[:,3] = 1
print(cola)
points = distTest1.toNpArrayCopy()
min_slider = np.min(distances)
max_slider = np.max(distances)
slider_step = (max_slider-min_slider)/10
color_data = cola  # Random color data in (r, g, b, alpha) format


# Define colormap from blue to green to red
cmap = matplotlib.cm.get_cmap('RdYlBu_r')
norm = mcolors.Normalize(vmin=np.min(distances), vmax=np.max(distances))  # Normalize distance values
color_palette = [mcolors.rgb2hex(cmap(norm(d))) for d in distances]  # Convert distances to colors

color_values = np.linspace(0, 1, len(color_palette))

# Create the color bar trace with constant values
color_bar_trace = go.Scatter(
    x=[None],  # Dummy x-values
    y=[None],  # Dummy y-values
    mode='markers',
    marker=dict(
        colorscale='RdYlBu_r',
        color=color_values,
        showscale=True,
        colorbar=dict(
            title='Distance',

        )
    ),
    showlegend=False,

)

# Initialize Dash app
app = dash.Dash(__name__)

# Define app layout
app.layout = html.Div([
    dcc.Graph(
        id='3d-scatter-plot',style={'width': '150vh', 'height': '95vh'},
        config={
            'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False, 'editable': False,#'orbitRotation' : True
                },
        figure={
            'data': [
                go.Scatter3d(
                    x=points[:,0],
                    y=points[:,1],
                    z=points[:,2],
                    mode='markers',
                    marker=dict(
                        size=1,
                        #color=distances,
                        opacity=0.8,
                        color = color_palette,
                        colorscale='RdYlBu_r',  # Set colorscale
                        showscale=True,  # Show color scale,
                        colorbar=dict(title='Distance')  # Colorbar title
                    )
                ),
                color_bar_trace
            ],
            'layout': go.Layout(
                title='3D Scatter Plot',
                scene=dict(
                    xaxis=dict(title='X', showgrid=False, showticklabels=False, zeroline=False, showline=False),
                    yaxis=dict(title='Y', showgrid=False, showticklabels=False, zeroline=False, showline=False),
                    zaxis=dict(title='Z')
                )
            )
        }
      ),
    dcc.Slider(
        id='distance-slider',
        min=min_slider,
        max=max_slider,
        step=slider_step,
        value=max_slider

    )
])
@app.callback(
    Output('3d-scatter-plot', 'figure'),
  # Include state to retrieve current camera position
    [Input('distance-slider', 'value')],
    [State('3d-scatter-plot', 'figure')],
    prevent_initial_call=True
)
def update_figure(selected_distance,existing_figure):

    #print("i am here")
    # Apply color changes
    updated_palette = [color if d <= selected_distance else "#000000" for d, color in zip(distances, color_palette)]
    # Update scatter plot
    new_figure ={
            'data': [
                go.Scatter3d(
                    x=points[:,0],
                    y=points[:,1],
                    z=points[:,2],
                    mode='markers',
                    marker=dict(
                        size=1,
                        #color=distances,
                        opacity=0.8,
                        color = updated_palette,
                        colorscale='RdYlBu_r',  # Set colorscale
                        showscale=True,  # Show color scale,
                        colorbar=dict(title='Distance')  # Colorbar title
                    )
                ),
                color_bar_trace
            ],
            'layout': go.Layout(
                title='3D Scatter Plot',
                scene=dict(
                    xaxis=dict(title='X'),
                    yaxis=dict(title='Y'),
                    zaxis=dict(title='Z'),
                    camera=existing_figure['layout']['scene']['camera']  # Retrieve and set current camera position
                )
            )
        }

    #print("Updated Figure:", new_figure)

    return new_figure
# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True,port = 5454)