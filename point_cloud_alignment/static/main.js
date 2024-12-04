

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



//const loader = new OBJLoader();

document.addEventListener('DOMContentLoaded', function() {
    init();
    animate();
});

function init() {
    scene = new THREE.Scene();

    camera = new THREE.PerspectiveCamera(70, window.innerWidth/window.innerHeight, 0.01, 1000);
    camera.position.set(0, 0, 30); // Bring the camera closer

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
    const axesHelper = new THREE.AxesHelper(10);
    scene.add(axesHelper);
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
    raycaster.params.Points.threshold = 0.1;

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

        // Add a small sphere at the intersection point for debugging
        const sphereGeometry = new THREE.SphereGeometry(0.1, 16, 16);
        const sphereMaterial = new THREE.MeshBasicMaterial({ color: 0xffff00 });
        const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
        sphere.position.copy(intersect.point);
        scene.add(sphere);

        console.log(`Intersection found at index ${idx}:`, intersect.point);
    } else {
        console.warn("No intersection found!");
    }

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


            const localPoint = new THREE.Vector3(
                positionAttribute.getX(idx),
                positionAttribute.getY(idx),
                positionAttribute.getZ(idx)
            );

            // Convert local point to global coordinates
            const globalPoint = localPoint.clone().applyMatrix4(pointCloud.matrixWorld);
            points.push([globalPoint.x, globalPoint.y, globalPoint.z]);
            //points.push([x, y, z]);

            // Change color to selection color (yellow)
            pointCloud.geometry.attributes.color.setXYZ(idx, 1, 1, 0); // Yellow
        
            
            //pointCloud.geometry.attributes.size.setSize(5)
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
        size: pointCloud.material.size * 5, // Increase size by a factor (e.g., 2)
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
        const extension = file.name.split('.').pop().toLowerCase();

        let geometry;

        if (extension === 'ply') {
            const loader = new THREE.PLYLoader();
            geometry = loader.parse(contents);
        } else if (extension === 'obj') {
            const loader = new THREE.OBJLoader();
            const obj = loader.parse(contents);

            // Extract vertices from the OBJ object
            geometry = new THREE.BufferGeometry();

            let positions = [];

            obj.traverse(function(child) {
                if (child.isMesh) {
                    const positionAttribute = child.geometry.attributes.position;
                    positions.push(...positionAttribute.array);
                }
            });

            positions = new Float32Array(positions);
            geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        } else {
            alert('Unsupported file format. Please select a PLY or OBJ file.');
            return;
        }
        //alignToGlobalFrame(geometry);

        //console.log('Source Points:', sourcePointCloud.geometry.attributes.position.array);
        //console.log('Target Points:', targetPointCloud.geometry.attributes.position.array);


        
        // Proceed with the rest of your code to process the geometry
        processLoadedGeometry(geometry, isSource);


    };
    reader.readAsArrayBuffer(file);
}

function processLoadedGeometry(geometry, isSource) {

    if (!geometry || !geometry.attributes || !geometry.attributes.position) {
        console.error('Invalid geometry:', geometry);
        return;
    }
    console.log(geometry.attributes.position.array);
    //alignToGlobalFrame(geometry);
    geometry.computeBoundingBox();
    const positions = geometry.attributes.position.array;
    console.log('First 10 Aligned Points:', positions.slice(0, 30));
    console.log('Aligned Bounding Box:', geometry.boundingBox);

    if (!geometry || !geometry.attributes.position || geometry.attributes.position.count === 0) {
        console.error('Geometry is empty or corrupted.');
        return;
    }

    // const testGeometry = new THREE.BoxGeometry(1, 1, 1);
    // alignToGlobalFrame(testGeometry);
    // console.log('Test Geometry Aligned Bounding Box:', testGeometry.boundingBox);
    
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
    pointCloud.renderOrder = isSource ? 1 : 2; // Source will render first, Target will render second
    if (!isSource) {
        material.depthTest = false; // Ensures target points are always interactive
    }
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

function alignToGlobalFrame(geometry) {
    const positions = geometry.attributes.position.array;
    const points = [];

    // Extract points from the geometry and filter out invalid ones
    for (let i = 0; i < positions.length; i += 3) {
        const x = positions[i];
        const y = positions[i + 1];
        const z = positions[i + 2];

        if (isNaN(x) || isNaN(y) || isNaN(z) || !isFinite(x) || !isFinite(y) || !isFinite(z)) {
            console.warn(`Corrupted point detected: (${x}, ${y}, ${z})`);
            continue;
        }
        points.push([x, y, z]);
    }

    // Compute mean of the points
    const mean = [0, 0, 0];
    for (const point of points) {
        mean[0] += point[0];
        mean[1] += point[1];
        mean[2] += point[2];
    }
    mean[0] /= points.length;
    mean[1] /= points.length;
    mean[2] /= points.length;

    // Center the points around the mean
    const centeredPoints = points.map(([x, y, z]) => [x - mean[0], y - mean[1], z - mean[2]]);

    // Compute covariance matrix
    const covariance = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (const [x, y, z] of centeredPoints) {
        covariance[0][0] += x * x;
        covariance[0][1] += x * y;
        covariance[0][2] += x * z;
        covariance[1][0] += y * x;
        covariance[1][1] += y * y;
        covariance[1][2] += y * z;
        covariance[2][0] += z * x;
        covariance[2][1] += z * y;
        covariance[2][2] += z * z;
    }

    // Normalize covariance matrix
    for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
            covariance[i][j] /= points.length;
        }
    }

    // Compute eigenvalues and eigenvectors
    const eigenResult = math.eigs(covariance);
    const eigenvectors = eigenResult.vectors;

    // Rotate the point cloud to align with the global frame
    for (let i = 0; i < positions.length; i += 3) {
        const point = [positions[i] - mean[0], positions[i + 1] - mean[1], positions[i + 2] - mean[2]];
        const alignedPoint = math.multiply(eigenvectors, point);
        positions[i] = alignedPoint[0];
        positions[i + 1] = alignedPoint[1];
        positions[i + 2] = alignedPoint[2];
    }

    geometry.attributes.position.needsUpdate = true;
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
    

    console.log('Source Points:', sourcePoints);
    console.log('Target Points:', targetPoints);

    console.log('Source Bounding Box:', sourcePointCloud.geometry.boundingBox);
    console.log('Target Bounding Box:', targetPointCloud.geometry.boundingBox);

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
function getGlobalPoint(localPoint, object) {
    const globalPoint = new THREE.Vector3(localPoint.x, localPoint.y, localPoint.z);
    globalPoint.applyMatrix4(object.matrixWorld); // Convert to global coordinates
    return globalPoint;
}
function downloadTransformedPCD() {
    window.location.href = '/download_transformed_pcd';
}

function toggleSelectionMode() {
    selectingSource = !selectingSource;
    // if (selectingSource) {
    //     if (sourcePointCloud) sourcePointCloud.visible = true; // Show source
    //     if (targetPointCloud) targetPointCloud.visible = false; // Hide target
    // } else {
    //     if (sourcePointCloud) sourcePointCloud.visible = false; // Hide source
    //     if (targetPointCloud) targetPointCloud.visible = true; // Show target
    // }
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
