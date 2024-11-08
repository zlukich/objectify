// Variables to store scene, camera, renderer
let scene, camera, renderer, controls;
let sourcePoints = [], targetPoints = [];
let sourceSelections = [], targetSelections = [];
let sourcePointCloud, targetPointCloud, transformedPointCloud;
let raycaster = new THREE.Raycaster();
let mouse = new THREE.Vector2();
let selectingSource = true; // Indicates whether we're selecting source or target points
let highlightPointCloud = null;
let previousHoverIndex = null;
let previousHoverPointCloud = null;

document.addEventListener('DOMContentLoaded', function() {
    init();
    animate();
});

function init() {
    scene = new THREE.Scene();

    camera = new THREE.PerspectiveCamera(70, window.innerWidth/window.innerHeight, 0.01, 1000);
    camera.position.z = 5;

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    // Customize controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);

    // Set controls to use modifier keys
    controls.enableRotate = true;
    controls.enableZoom = true;
    controls.enablePan = true;

    // Configure mouse buttons and modifier keys
    controls.mouseButtons = {
        LEFT: THREE.MOUSE.LEFT,   // Rotate
        MIDDLE: THREE.MOUSE.MIDDLE, // Zoom
        RIGHT: THREE.MOUSE.RIGHT   // Pan
    };

    controls.keyPanSpeed = 7.0; // Increase pan speed for keyboard controls

    // Add event listener for Shift key to enable panning
    window.addEventListener('keydown', function(event) {
        if (event.shiftKey) {
            controls.enablePan = true;
            controls.enableRotate = false;
        }
    }, false);

    window.addEventListener('keyup', function(event) {
        if (!event.shiftKey) {
            controls.enablePan = false;
            controls.enableRotate = true;
        }
    }, false);

    // Adjust raycaster threshold
    raycaster.params.Points.threshold = 0.05;

    // Event listeners
    window.addEventListener('resize', onWindowResize, false);
    renderer.domElement.addEventListener('click', onMouseClick, false);
    renderer.domElement.addEventListener('mousemove', onMouseMove, false);

    // File upload handlers
    document.getElementById('source-file').addEventListener('change', onSourceFileSelected, false);
    document.getElementById('target-file').addEventListener('change', onTargetFileSelected, false);

    document.getElementById('align-button').addEventListener('click', alignPointClouds, false);
    document.getElementById('download-button').addEventListener('click', downloadTransformedPCD, false);

    // Toggle selection mode using the button
    document.getElementById('toggle-selection').addEventListener('click', toggleSelectionMode, false);

    // Set initial button style and text
    updateSelectionModeIndicator();

    // Add lights
    const ambientLight = new THREE.AmbientLight(0x404040, 1.5); // Soft white light
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
    directionalLight.position.set(1, 1, 1).normalize();
    scene.add(directionalLight);
}


function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();

    renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
    requestAnimationFrame(animate);

    controls.update();

    renderer.render(scene, camera);
}

function onMouseClick(event) {
    event.preventDefault();

    // Reset hover color if necessary
    if (previousHoverPointCloud && previousHoverIndex !== null) {
        resetPointColor(previousHoverPointCloud, previousHoverIndex);
        previousHoverIndex = null;
        previousHoverPointCloud = null;
    }

    const canvasBounds = renderer.domElement.getBoundingClientRect();

    mouse.x = ((event.clientX - canvasBounds.left) / canvasBounds.width) * 2 - 1;
    mouse.y = - ((event.clientY - canvasBounds.top) / canvasBounds.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);

    // Use selectingSource to determine which point cloud to interact with
    const pointCloud = selectingSource ? sourcePointCloud : targetPointCloud;

    if (!pointCloud) return; // If point cloud is not loaded yet

    const intersects = raycaster.intersectObject(pointCloud);

    if (intersects.length > 0) {
        const intersect = intersects[0];
        const idx = intersect.index;

        const selections = selectingSource ? sourceSelections : targetSelections;
        const points = selectingSource ? sourcePoints : targetPoints;
        const countSpan = selectingSource ? document.getElementById('source-count') : document.getElementById('target-count');

        if (!selections.includes(idx)) {
            selections.push(idx);
            const positionAttribute = pointCloud.geometry.attributes.position;
            const x = positionAttribute.getX(idx);
            const y = positionAttribute.getY(idx);
            const z = positionAttribute.getZ(idx);
            points.push([x, y, z]);

            // Change color to selection color (yellow)
            pointCloud.geometry.attributes.color.setXYZ(idx, 1, 1, 0); // Yellow
        } else {
            // Deselect point
            const selIndex = selections.indexOf(idx);
            selections.splice(selIndex, 1);
            points.splice(selIndex, 1);

            // Reset color to original color
            const originalColor = getOriginalColor(pointCloud, idx);
            pointCloud.geometry.attributes.color.setXYZ(idx, originalColor.r, originalColor.g, originalColor.b);
        }

        pointCloud.geometry.attributes.color.needsUpdate = true;
        countSpan.textContent = selections.length;

        if (sourceSelections.length >= 4 && targetSelections.length >= 4) {
            document.getElementById('align-button').disabled = false;
        } else {
            document.getElementById('align-button').disabled = true;
        }
    }
}

function onMouseMove(event) {
    event.preventDefault();

    const canvasBounds = renderer.domElement.getBoundingClientRect();

    mouse.x = ((event.clientX - canvasBounds.left) / canvasBounds.width) * 2 - 1;
    mouse.y = - ((event.clientY - canvasBounds.top) / canvasBounds.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);

    // Use selectingSource to determine which point cloud to interact with
    const pointCloud = selectingSource ? sourcePointCloud : targetPointCloud;

    if (!pointCloud) return;

    const intersects = raycaster.intersectObject(pointCloud);

    if (intersects.length > 0) {
        const intersect = intersects[0];
        const idx = intersect.index;

        if (previousHoverIndex !== idx || previousHoverPointCloud !== pointCloud) {
            if (previousHoverPointCloud && previousHoverIndex !== null) {
                resetPointColor(previousHoverPointCloud, previousHoverIndex);
            }

            highlightPoint(pointCloud, idx);

            previousHoverIndex = idx;
            previousHoverPointCloud = pointCloud;
        }
    } else {
        if (previousHoverPointCloud && previousHoverIndex !== null) {
            resetPointColor(previousHoverPointCloud, previousHoverIndex);
            previousHoverIndex = null;
            previousHoverPointCloud = null;
        }
    }
}


function highlightPoint(pointCloud, idx) {
    // Change color to hover color (cyan)
    pointCloud.geometry.attributes.color.setXYZ(idx, 0, 1, 1); // Cyan
    pointCloud.geometry.attributes.color.needsUpdate = true;

    // Create a larger point at the highlighted point's position
    const positionAttribute = pointCloud.geometry.attributes.position;
    const x = positionAttribute.getX(idx);
    const y = positionAttribute.getY(idx);
    const z = positionAttribute.getZ(idx);

    // Create geometry for the highlighted point
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array([x, y, z]);
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    // Create a material with a larger size
    const material = new THREE.PointsMaterial({
        size: pointCloud.material.size * 2, // Increase size by a factor (e.g., 2)
        color: 0xffff00, // Highlight color (yellow)
        sizeAttenuation: true
    });

    // Create the point cloud for the highlighted point
    highlightPointCloud = new THREE.Points(geometry, material);

    // Add to the scene
    scene.add(highlightPointCloud);

    // Change cursor style
    renderer.domElement.style.cursor = 'pointer';
}


function resetPointColor(pointCloud, idx) {
    const selections = (pointCloud === sourcePointCloud) ? sourceSelections : targetSelections;
    const isSelected = selections.includes(idx);

    if (isSelected) {
        // If the point is selected, set it to selection color (yellow)
        pointCloud.geometry.attributes.color.setXYZ(idx, 1, 1, 0); // Yellow
    } else {
        // Reset to original color
        const originalColor = getOriginalColor(pointCloud, idx);
        pointCloud.geometry.attributes.color.setXYZ(idx, originalColor.r, originalColor.g, originalColor.b);
    }

    pointCloud.geometry.attributes.color.needsUpdate = true;

    // Remove the highlight point cloud if it exists
    if (highlightPointCloud) {
        scene.remove(highlightPointCloud);
        highlightPointCloud.geometry.dispose();
        highlightPointCloud.material.dispose();
        highlightPointCloud = null;
    }

    // Reset cursor style
    renderer.domElement.style.cursor = 'default';
}


function getOriginalColor(pointCloud, idx) {
    // Retrieve the original color from the color attribute
    const colorAttribute = pointCloud.geometry.attributes.originalColor;
    const r = colorAttribute.getX(idx);
    const g = colorAttribute.getY(idx);
    const b = colorAttribute.getZ(idx);
    return { r: r, g: g, b: b };
}

function onSourceFileSelected(event) {
    const file = event.target.files[0];
    if (file) {
        loadPointCloud(file, true);
    }
}

function onTargetFileSelected(event) {
    const file = event.target.files[0];
    if (file) {
        loadPointCloud(file, false);
    }
}

function loadPointCloud(file, isSource) {
    const reader = new FileReader();
    reader.onload = function(e) {
        const contents = e.target.result;
        const loader = new THREE.PLYLoader();
        const geometry = loader.parse(contents);

        // Compute bounding box to get object size
        geometry.computeBoundingBox();
        const bbox = geometry.boundingBox;
        const size = new THREE.Vector3();
        bbox.getSize(size);
        const maxDimension = Math.max(size.x, size.y, size.z);

        // Adjust point size based on object size
        const basePointSize = 0.05; // You can adjust this value
        const scalingFactor = 10; // You can adjust this value
        const pointSize = basePointSize * (maxDimension / scalingFactor);

        // Assign a single color to all points
        const color = new THREE.Color(isSource ? 0xff0000 : 0x0000ff); // Red for source, blue for target

        const colors = new Float32Array(geometry.attributes.position.count * 3);
        const originalColors = new Float32Array(geometry.attributes.position.count * 3);
        for (let i = 0; i < geometry.attributes.position.count; i++) {
            colors[i * 3] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;

            // Store original colors for resetting later
            originalColors[i * 3] = color.r;
            originalColors[i * 3 + 1] = color.g;
            originalColors[i * 3 + 2] = color.b;
        }
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geometry.setAttribute('originalColor', new THREE.BufferAttribute(originalColors, 3));

        // Create PointsMaterial
        const material = new THREE.PointsMaterial({
            size: pointSize,
            vertexColors: true,
            sizeAttenuation: true
        });

        const pointCloud = new THREE.Points(geometry, material);

        if (isSource) {
            if (sourcePointCloud) scene.remove(sourcePointCloud);
            sourcePointCloud = pointCloud;
            scene.add(sourcePointCloud);
        } else {
            if (targetPointCloud) scene.remove(targetPointCloud);
            targetPointCloud = pointCloud;
            scene.add(targetPointCloud);
        }

        // Upload point cloud to backend
        uploadPointClouds();
    };
    reader.readAsArrayBuffer(file);
}

function uploadPointClouds() {
    const sourceFile = document.getElementById('source-file').files[0];
    const targetFile = document.getElementById('target-file').files[0];

    if (sourceFile && targetFile) {
        const formData = new FormData();
        formData.append('source', sourceFile);
        formData.append('target', targetFile);

        fetch('/upload_pointclouds', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => {
                    throw new Error(err.error || 'Error uploading point clouds');
                }).catch(() => {
                    throw new Error('Error uploading point clouds');
                });
            }
            return response.json();
        })
        .then(data => {
            console.log(data.status);
        })
        .catch(error => {
            console.error('Error uploading point clouds:', error);
            alert('Error uploading point clouds: ' + error.message);
        });
    }
}

// Add event listener for the "Clear Point Clouds" button
document.getElementById('clear-pointclouds-button').addEventListener('click', clearPointClouds, false);

function clearPointClouds() {
    // Remove source point cloud from the scene
    if (sourcePointCloud) {
        scene.remove(sourcePointCloud);
        sourcePointCloud.geometry.dispose();
        sourcePointCloud.material.dispose();
        sourcePointCloud = null;
    }

    // Remove target point cloud from the scene
    if (targetPointCloud) {
        scene.remove(targetPointCloud);
        targetPointCloud.geometry.dispose();
        targetPointCloud.material.dispose();
        targetPointCloud = null;
    }

    // Remove transformed point cloud from the scene
    if (transformedPointCloud) {
        scene.remove(transformedPointCloud);
        transformedPointCloud.geometry.dispose();
        transformedPointCloud.material.dispose();
        transformedPointCloud = null;
    }

    // Clear selections and points
    sourceSelections = [];
    targetSelections = [];
    sourcePoints = [];
    targetPoints = [];
    previousHoverIndex = null;
    previousHoverPointCloud = null;

    // Reset selection counts
    document.getElementById('source-count').textContent = '0';
    document.getElementById('target-count').textContent = '0';

    // Disable align and download buttons
    document.getElementById('align-button').disabled = true;
    document.getElementById('download-button').disabled = true;

    // Clear file input fields (optional)
    document.getElementById('source-file').value = '';
    document.getElementById('target-file').value = '';
    if (highlightPointCloud) {
        scene.remove(highlightPointCloud);
        highlightPointCloud.geometry.dispose();
        highlightPointCloud.material.dispose();
        highlightPointCloud = null;
    }
}


document.getElementById('clear-selections-button').addEventListener('click', clearSelections, false);

function clearSelections() {
    // Clear source selections
    if (sourcePointCloud) {
        sourceSelections.forEach(idx => {
            // Reset color to original color
            const originalColor = getOriginalColor(sourcePointCloud, idx);
            sourcePointCloud.geometry.attributes.color.setXYZ(idx, originalColor.r, originalColor.g, originalColor.b);
        });
        sourcePointCloud.geometry.attributes.color.needsUpdate = true;
    }
    sourceSelections = [];
    sourcePoints = [];
    document.getElementById('source-count').textContent = '0';

    // Clear target selections
    if (targetPointCloud) {
        targetSelections.forEach(idx => {
            // Reset color to original color
            const originalColor = getOriginalColor(targetPointCloud, idx);
            targetPointCloud.geometry.attributes.color.setXYZ(idx, originalColor.r, originalColor.g, originalColor.b);
        });
        targetPointCloud.geometry.attributes.color.needsUpdate = true;
    }
    targetSelections = [];
    targetPoints = [];
    document.getElementById('target-count').textContent = '0';

    // Disable the align button
    document.getElementById('align-button').disabled = true;
    if (highlightPointCloud) {
        scene.remove(highlightPointCloud);
        highlightPointCloud.geometry.dispose();
        highlightPointCloud.material.dispose();
        highlightPointCloud = null;
    }
}

function alignPointClouds() {
    if (sourcePoints.length !== targetPoints.length) {
        alert('The number of selected source points and target points must be the same.');
        return;
    }
    const data = {
        source_points: sourcePoints,
        target_points: targetPoints
    };
    

    fetch('/align_pointclouds', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => {
                throw new Error(err.error || 'Error aligning point clouds');
            }).catch(() => {
                throw new Error('Error aligning point clouds');
            });
        }
        return response.json();
    })
    .then(data => {
        // Remove old transformed point cloud if exists
        if (transformedPointCloud) scene.remove(transformedPointCloud);

        const transformedPoints = data.transformed_points;
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(transformedPoints.length * 3);

        for (let i = 0; i < transformedPoints.length; i++) {
            positions[i * 3] = transformedPoints[i][0];
            positions[i * 3 + 1] = transformedPoints[i][1];
            positions[i * 3 + 2] = transformedPoints[i][2];
        }

        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

        // Compute bounding box to get object size
        geometry.computeBoundingBox();
        const bbox = geometry.boundingBox;
        const size = new THREE.Vector3();
        bbox.getSize(size);
        const maxDimension = Math.max(size.x, size.y, size.z);

        // Adjust point size based on object size
        const basePointSize = 0.05; // You can adjust this value
        const scalingFactor = 10; // You can adjust this value
        const pointSize = basePointSize * (maxDimension / scalingFactor);

        // Assign a single color (green) to all points
        const color = new THREE.Color(0x00ff00); // Green for transformed point cloud

        const colors = new Float32Array(transformedPoints.length * 3);
        for (let i = 0; i < transformedPoints.length; i++) {
            colors[i * 3] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;
        }
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        // Create PointsMaterial
        const material = new THREE.PointsMaterial({
            size: pointSize,
            vertexColors: true,
            sizeAttenuation: true
        });

        transformedPointCloud = new THREE.Points(geometry, material);
        scene.add(transformedPointCloud);

        document.getElementById('download-button').disabled = false;
    })
    .catch(error => {
        console.error('Error aligning point clouds:', error);
        alert('Error aligning point clouds: ' + error.message);
    });
}

function downloadTransformedPCD() {
    window.location.href = '/download_transformed_pcd';
}

function toggleSelectionMode() {
    selectingSource = !selectingSource;
    updateSelectionModeIndicator();
}

function updateSelectionModeIndicator() {
    const toggleButton = document.getElementById('toggle-selection');
    const selectionModeText = document.getElementById('selection-mode');
    if (selectingSource) {
        toggleButton.textContent = 'Switch to Target Selection';
        toggleButton.style.backgroundColor = 'red';
        selectionModeText.textContent = 'Currently Selecting: Source Points';
    } else {
        toggleButton.textContent = 'Switch to Source Selection';
        toggleButton.style.backgroundColor = 'blue';
        selectionModeText.textContent = 'Currently Selecting: Target Points';
    }
}
