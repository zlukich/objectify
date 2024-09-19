import argparse
from pathlib import Path
from rembg import remove, new_session
import numpy as np
from PIL import Image
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Remove background from images.')
parser.add_argument('--model', type=str, choices=['isnet-general-use', 'sam',"u2net","birefnet-general","birefnet-general-lite"], default='isnet-general-use', help='Model to use for background removal.')
parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing the images.')
args = parser.parse_args()

model_name = args.model
session = new_session(model_name)

input_dir = Path(args.input_dir)
output_dir = input_dir / 'rembg'
output_dir.mkdir(parents=True, exist_ok=True)

image_files = list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpg')) +list(input_dir.glob('*.jpeg'))
print(image_files)
num_images = len(image_files)

if num_images == 0:
    print("No images found in the input directory.")
    exit()



input_points = []
input_labels = []
if model_name == "sam":
    for file in image_files:
        # Open the first image to get dimensions
        image = Image.open(file)
        width, height = image.size
        center_point = [width / 2, height / 2]
        input_labels.append(1)
        input_points.append(center_point)
    
    input_labels = np.array(input_labels)
    input_points = np.array(input_points)
        

for file in image_files:
    input_path = str(file)
    output_path = str(output_dir / (file.stem + ".png"))

    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input_data = i.read()

            if model_name == "sam":
                output = remove(input_data, session=session, sam_prompt=[{"type": "point", "data": center_point, "label": 1}], post_process_mask=True)
            else:
                output = remove(input_data, session=session, post_process_mask=True)

            o.write(output)

