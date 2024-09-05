import argparse
from pathlib import Path
from rembg import remove, new_session
import numpy as np
from PIL import Image
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Remove background from images.')
parser.add_argument('--model', type=str, choices=['isnet-general-use', 'sam'], default='isnet-general-use', help='Model to use for background removal.')
parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing the images.')
args = parser.parse_args()

model_name = args.model
session = new_session(model_name)

input_dir = Path(args.input_dir)
output_dir = input_dir / 'rembg'
output_dir.mkdir(parents=True, exist_ok=True)

image_files = list(input_dir.glob('*.jpg'))
num_images = len(image_files)

if num_images == 0:
    print("No images found in the input directory.")
    exit()

# Open the first image to get dimensions
first_image = Image.open(image_files[0])
width, height = first_image.size
center_point = [width / 2, height / 2]

if model_name == "sam":
    input_labels = np.array([1] * num_images)
    input_points = np.array([center_point] * num_images)

for file in image_files:
    input_path = str(file)
    output_path = str(output_dir / (file.stem + ".out.png"))

    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input_data = i.read()

            if model_name == "sam":
                output = remove(input_data, session=session, sam_prompt=[{"type": "point", "data": center_point, "label": 1}], post_process_mask=True)
            else:
                output = remove(input_data, session=session, post_process_mask=True)

            o.write(output)

