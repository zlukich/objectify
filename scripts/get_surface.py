import trimesh
import mesh_to_sdf as mts
import argparse
import time
def write_obj(vertices, output_obj_path='surface_points.obj'):
    with open(output_obj_path, 'w') as obj_file:
        for vertex in vertices:
            obj_file.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')

    print(f'OBJ file saved to {output_obj_path}', flush = True)

parser = argparse.ArgumentParser(       
    prog='Get Surface',
    description='Receives only surface points from object represented as mesh(can be non-watertight mesh)',
    epilog='You can!'
)
start = time.time()
parser.add_argument('--mesh_path')
parser.add_argument('--num_points', type = int, default = 1e6 )
parser.add_argument("--out_name",type = str, default = "surface_points.obj")
args = parser.parse_args()
file = args.mesh_path
num_points = args.num_points
out_name = args.out_name

mesh = trimesh.load_mesh(file)  
print(f"Starting the surface generation", flush = True)
surface = mts.get_surface_point_cloud(mesh,
                                      surface_point_method='scan',
                                      bounding_radius=1,
                                      scan_count=100,
                                      scan_resolution=400,
                                      sample_point_count=6000000,
                                      calculate_normals=False)

points = surface.get_random_surface_points(count = num_points)

write_obj(points,output_obj_path=out_name)
end = time.time()

print(f"Surface Generation took {end - start} seconds",flush = True)