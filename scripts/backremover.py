import argparse
from pathlib import Path
from rembg import remove, new_session
import numpy as np
from PIL import Image
import os
import onnxruntime as ort
print(ort.get_device())  # Should return 'GPU'

import time
start = time.time()

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Remove background from images.')
parser.add_argument('--model', type=str, choices=['sam','isnet-general-use' ,"u2net","birefnet-general","birefnet-general-lite"], default='isnet-general-use', help='Model to use for background removal.')
parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing the images.')
parser.add_argument('--output_dir', type=str, default="rembg", help='Output directory name containing the processed images.')
parser.add_argument('--post_process_mask', type=bool, default=True, help='Perform mask post processing')


args = parser.parse_args()

model_name = args.model
session = new_session(model_name)

#print(session.inner_session.get_providers())

input_dir = Path(args.input_dir)
post_process_mask = args.post_process_mask
output_dir = input_dir / args.output_dir
output_dir.mkdir(parents=True, exist_ok=True)

image_files = list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpg')) +list(input_dir.glob('*.jpeg')) + list(input_dir.glob('*.bmp'))
print(image_files,flush = True)
num_images = len(image_files)

if num_images == 0:
    print("No images found in the input directory.",flush = True)
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
    print(f"Processing image {str(file)}",flush = True)
    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input_data = i.read()
            
            if model_name == "sam":
                output = remove(input_data, session=session, sam_prompt=[{"type": "point", "data": center_point, "label": 1}], post_process_mask=post_process_mask)
            else:
                output = remove(input_data, session=session, post_process_mask=True, alpha_matting= post_process_mask)

            o.write(output)


print('Background removal took', time.time()-start, 'seconds.', flush=True)