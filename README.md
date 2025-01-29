
---

# 3D Reconstruction of Additive Manufactured Workpieces for Quality Control: Integrating Photogrammetry, Neural Radiance Fields, and Deep Learning

## Abstract from this work's Paper

This thesis presents the development of a novel quality control tool for additive manufacturing
workpieces, leveraging advanced 3D reconstruction and machine vision techniques. The sys-
tem integrates state-of-the-art methods, including photogrammetry, Neural Radiance Fields
(NeRF), and deep learning, to achieve precise reconstructions and automated defect detec-
tion. The proposed pipeline enables high-speed 3D reconstruction with sub-minute execution
times, delivering superior geometric accuracy compared to traditional photogrammetric ap-
proaches. Validation is conducted using a custom-developed dataset including diverse object
types, ensuring robustness across varying geometries and material properties. Key perfor-
mance metrics include dimension recovery accuracy and average Chamfer and Hausdorff
distances, offering comprehensive measures of reconstruction quality. Notably, the developed
pipeline reduces the average Chamfer distance by 226% and the average Hausdorff distance
by an impressive 1590% (lower values indicate better performance), while achieving total
reconstruction times of just a few minutes and significantly outperforming traditional pho-
togrammetry, which averages 31.16 minutes per reconstruction. Additionally, a quality con-
trol application is developed, featuring a user-friendly interface for visualization, comparison,
and defect analysis, making it highly applicable for industrial use. This work demonstrates
the potential of combining cutting-edge 3D reconstruction techniques with deep learning to
optimize and automate quality assurance processes.


---

## Project Folders Description

This repository is organized into several key folders:

- **lib**: Includes utility functions for:
  - **Calibration**: Functions for handling Charuco board calibration.
  - **JSON Generation**: Scripts for creating and manipulating JSON files for camera poses and other metadata.
  - **Geometry Functions**: Helper functions for 3D geometric calculations.
- **scripts**: Python scripts called by the Node-RED server to perform tasks such as:
  - Image acquisition.
  - Pose estimation.
  - Mesh extraction and comparison.
  - Initiating reconstruction scripts.
- **node-red**: The main application folder. Contains Node-RED flows and configuration files. The application serves as a user interface for data visualization and quality control.
- **environment.yml**: Environment configuration file for setting up the required Python dependencies.

---

## Parameter Description

The following are key parameters used in the project:

1. **Charuco Board**:
   - **ARUCO_DICT**: `cv2.aruco.DICT_6X6_250` (Defines the marker dictionary used for calibration).
   - **SQUARES_VERTICALLY**: 13.
   - **SQUARES_HORIZONTALLY**: 9.
   - **SQUARE_LENGTH**: 0.03m.
   - **MARKER_LENGTH**: 0.02m.

2. **Marching Cubes Parameters**:
   - **mc_resolution**: Defines the resolution of the grid for mesh extraction. Higher values produce finer meshes.
   - **mc_threshold**: Threshold for determining surface boundaries in the scalar field.

3. **Application Parameters**:
   - Default Node-RED server port: `9800`.
   - Path to save processed results and intermediate files.

---

## Installation

To set up the environment and install required dependencies:

1. **Create the Python Environment**:
   ```bash
   conda env create -f environment.yml
   conda activate <environment-name>
   ```
2. **Install Instant-NGP**:
   Follow the instructions provided in the official repository: [instant-ngp](https://github.com/NVlabs/instant-ngp).
3. **Install Colmap**:
   Download and install Colmap from [Colmap's official website](https://colmap.github.io/).
4. **Install Node-RED**:
   Follow the local installation instructions from [Node-RED's website](https://nodered.org/docs/getting-started/local).

---

## Example of Usage

The workflow for obtaining and processing data:

1. **Prepare the Environment**:
   - Activate the environment: `conda activate <environment-name>`.
   - Start the Node-RED server: 
     ```bash
     cd node-red
     node-red
     ```
   - Access the application via your web browser at `http://localhost:9800`.

2. **Data Acquisition**:
   - Use the interface to capture images with the connected camera.
   - Ensure the calibration files are ready for pose estimation.

3. **Pose Estimation**:
   - Use Colmap or Charuco tab for generating camera poses.

4. **Mesh Extraction**:
   - Run the reconstruction using API for instant-ngp algorithms with specified parameters.
    

---

## Data

The validation dataset used in this project can be downloaded [here](<insert-download-link>). The dataset contains images, calibration files, and ground truth meshes for various test objects.

---

## Important Remarks

1. **Best Practices**:
   - Always verify that the camera is calibrated before acquiring data. Misalignment in poses can result in inaccurate reconstructions.
   - Use consistent lighting conditions to ensure high-quality images for reconstruction.

2. **Common Issues**:
   - **Node-RED Server Not Starting**: Ensure the correct environment is activated and Node-RED is installed properly.
   - **Pose Estimation Errors**: Verify that the Charuco board is correctly placed in the scene and well-lit during image capture.
   - **Mesh Extraction Artifacts**: Fine-tune the `mc_resolution` and `mc_threshold` parameters to improve mesh quality.

---
