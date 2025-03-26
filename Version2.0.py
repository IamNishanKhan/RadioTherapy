import slicer
import os
import numpy as np
from PIL import Image
import nibabel as nib
import cv2

# Define the structures to include
included_structures = [
    "BODY",
    "URINARY BLADDER",
    "SMALL BOWEL",
    "RECTUM",
    "FEMORAL HEAD_RT",
    "FEMORAL HEAD_LT",
    "GTV",
    "CTV"
]

# Define the desired order
desired_order = [
    "BODY",
    "CTV",
    "GTV",
    "FEMORAL HEAD_RT",
    "FEMORAL HEAD_LT",
    "SMALL BOWEL",
    "URINARY BLADDER",
    "RECTUM"  # Top-most (drawn last)
]

# Define the standardized label mapping
standard_label_mapping = {
    "BODY": 1,
    "URINARY BLADDER": 2,
    "SMALL BOWEL": 3,
    "RECTUM": 4,
    "FEMORAL HEAD_RT": 5,
    "FEMORAL HEAD_LT": 6,
    "GTV": 7,
    "CTV": 8,
}

# RGB color mapping aligned with standard_label_mapping
label_to_rgb_color = {
    0: (0, 0, 0),          # Background (black)
    1: (0, 255, 0),        # Body (green) #00ff00
    2: (0, 255, 255),      # Urinary Bladder (cyan) #00ffff
    3: (255, 146, 153),    # Small Bowel (light pink) #ff9299
    4: (128, 64, 64),      # Rectum (dark brown) #804040
    5: (255, 255, 0),      # Femoral Head_RT (yellow) #ffff00
    6: (255, 255, 0),      # Femoral Head_LT (yellow) #ffff00
    7: (255, 60, 255),     # GTV (magenta) #ff3cff
    8: (55, 55, 255),      # CTV (blue) #3737ff
}

# Function to assign standardized label values to segments
def assign_standardized_labels(segmentation_node):
    segmentation = segmentation_node.GetSegmentation()
    segment_ids = segmentation.GetSegmentIDs()
    segment_name_to_id = {segmentation.GetSegment(segment_id).GetName(): segment_id for segment_id in segment_ids}
    
    for segment_name, label_value in standard_label_mapping.items():
        if segment_name in segment_name_to_id:
            segment_id = segment_name_to_id[segment_name]
            segmentation.GetSegment(segment_id).SetLabelValue(label_value)
            assigned_label = segmentation.GetSegment(segment_id).GetLabelValue()
            print(f"Assigned label {label_value} to '{segment_name}', verified as {assigned_label}.")
        else:
            print(f"Segment '{segment_name}' not found in the segmentation.")

# Function to check available structures and verify their geometry
def check_available_structures(segmentation_node):
    segmentation = segmentation_node.GetSegmentation()
    segment_ids = segmentation.GetSegmentIDs()
    segment_names = [segmentation.GetSegment(segment_id).GetName() for segment_id in segment_ids]
    
    print(f"Available structures in segmentation: {segment_names}")
    
    valid_segments = []
    for segment_id in segment_ids:
        segment = segmentation.GetSegment(segment_id)
        segment_name = segment.GetName()
        if segment_name not in included_structures:
            continue  # Skip structures not in the included list
        representation = segment.GetRepresentation("Binary labelmap")
        
        if representation:
            try:
                array = slicer.util.arrayFromSegmentBinaryLabelmap(segmentation_node, segment_id)
                unique_vals = np.unique(array)
                print(f"Segment '{segment_name}' binary labelmap unique values: {unique_vals}")
                if np.any(array > 0):
                    valid_segments.append(segment_name)
                    print(f"Segment '{segment_name}' has valid geometry and non-zero voxels.")
                else:
                    print(f"Segment '{segment_name}' has valid geometry but contains no non-zero voxels.")
            except Exception as e:
                print(f"Failed to extract array for segment '{segment_name}': {e}")
        else:
            print(f"Segment '{segment_name}' has no binary labelmap representation and will be skipped.")
    
    return valid_segments

# Function to reorder segments
def reorder_segments(segmentation_node, desired_order):
    segmentation = segmentation_node.GetSegmentation()
    segment_ids = segmentation.GetSegmentIDs()
    segment_name_to_id = {segmentation.GetSegment(segment_id).GetName(): segment_id for segment_id in segment_ids}
    
    ordered_segments = []
    for segment_name in desired_order:
        if segment_name in segment_name_to_id:
            segment_id = segment_name_to_id[segment_name]
            segment = segmentation.GetSegment(segment_id)
            ordered_segments.append((segment_name, segment))
    
    for segment_id in segment_ids:
        segmentation.RemoveSegment(segment_id)
    
    for segment_name, segment in ordered_segments:
        segmentation.AddSegment(segment, segment_name)
    
    print("Reordered segments successfully.")

# Function to export base DICOM images as PNGs for Axial, Coronal, and Sagittal views
def export_base_images(volume_node, output_dir):
    # Create output directories for Axial, Coronal, and Sagittal views
    axial_dir = os.path.join(output_dir, "original_images", "axial")
    coronal_dir = os.path.join(output_dir, "original_images", "coronal")
    sagittal_dir = os.path.join(output_dir, "original_images", "sagittal")
    for dir_path in [axial_dir, coronal_dir, sagittal_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Get the image data from the volume node
    image_data = slicer.util.arrayFromVolume(volume_node)
    print(f"Base image data shape: {image_data.shape}")
    
    # Normalize the image data to 0-255 for grayscale PNGs
    image_data = image_data.astype(np.float32)
    image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min()) * 255
    image_data = image_data.astype(np.uint8)
    
    # The image_data dimensions are (x, y, z) in RAS (Right-Anterior-Superior) orientation
    # - x: left to right
    # - y: posterior to anterior (back to front)
    # - z: inferior to superior (bottom to top)
    
    # --- Coronal View (up-down, slices along z-axis) ---
    # Coronal view shows the x-y plane (left-right, back-front) at each z level
    print(f"Generating axial base images (total: {image_data.shape[2]})...")
    for z in range(image_data.shape[2]):
        slice = image_data[:, :, z]  # Shape: (x, y)
        # Convert to PIL Image and resize
        base_image = Image.fromarray(slice, mode='L')  # 'L' mode for grayscale
        base_image = base_image.resize((512, 512), Image.NEAREST)
        base_image = base_image.transpose(Image.FLIP_TOP_BOTTOM)  # Flip vertically
        base_image.save(os.path.join(coronal_dir, f'slice_{z:03d}.png'))
        print(f"Saved axial base image {os.path.join(coronal_dir, f'slice_{z:03d}.png')}")
    
    # --- Axial View (right-left, slices along x-axis) ---
    # Axial view shows the y-z plane (back-front, bottom-top) at each x level
    axial_img = np.transpose(image_data, (0, 1, 2))  # (x, y, z)
    print(f"Generating coronal base images (total: {axial_img.shape[0]})...")
    for x in range(axial_img.shape[0]):
        slice = axial_img[x, :, :]  # Shape: (y, z)
        # Convert to PIL Image and resize
        base_image = Image.fromarray(slice, mode='L')
        base_image = base_image.resize((512, 512), Image.NEAREST)
        base_image.save(os.path.join(axial_dir, f'slice_{x:03d}.png'))
        print(f"Saved coronal base image {os.path.join(axial_dir, f'slice_{x:03d}.png')}")
    
    # --- Sagittal View (front-back, slices along y-axis) ---
    # Sagittal view shows the x-z plane (left-right, bottom-top) at each y level
    sagittal_img = np.transpose(image_data, (1, 0, 2))  # (y, x, z)
    print(f"Generating sagittal base images (total: {sagittal_img.shape[0]})...")
    for y in range(sagittal_img.shape[0]):
        slice = sagittal_img[y, :, :]  # Shape: (x, z)
        # Convert to PIL Image and resize
        base_image = Image.fromarray(slice, mode='L')
        base_image = base_image.resize((512, 512), Image.NEAREST)
        base_image = base_image.transpose(Image.FLIP_TOP_BOTTOM)  # Flip vertically
        base_image.save(os.path.join(sagittal_dir, f'slice_{y:03d}.png'))
        print(f"Saved sagittal base image {os.path.join(sagittal_dir, f'slice_{y:03d}.png')}")
    
    print(f"Converted base images to PNGs in {output_dir}")

# Function to convert segmentation to labelmap and export as NIfTI with custom labels (for masks)
def export_segmentation_as_nifti(segmentation_node, output_file_path):
    segmentation = segmentation_node.GetSegmentation()
    segment_ids = segmentation.GetSegmentIDs()
    segment_name_to_id = {segmentation.GetSegment(segment_id).GetName(): segment_id for segment_id in segment_ids}
    
    # Create a temporary labelmap volume node to get the geometry
    temp_labelmap_volume_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    success = slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
        segmentation_node, temp_labelmap_volume_node, slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY
    )
    if not success:
        slicer.mrmlScene.RemoveNode(temp_labelmap_volume_node)
        raise RuntimeError("Failed to convert segmentation to labelmap for geometry extraction.")
    
    # Get the dimensions and geometry from the temporary labelmap
    temp_labelmap_array = slicer.util.arrayFromVolume(temp_labelmap_volume_node)
    labelmap_shape = temp_labelmap_array.shape
    print(f"Mask labelmap shape: {labelmap_shape}")
    
    # Initialize the final labelmap array with zeros (background)
    final_labelmap_array = np.zeros(labelmap_shape, dtype=np.uint8)
    
    # Iterate over segments in the desired order and assign custom labels
    print("\nConstructing mask labelmap with custom labels:")
    for structure_name in desired_order:
        if structure_name not in segment_name_to_id:
            print(f"Segment '{structure_name}' not found, skipping.")
            continue
        segment_id = segment_name_to_id[structure_name]
        label_value = standard_label_mapping.get(structure_name)
        if label_value is None:
            print(f"No label value defined for '{structure_name}', skipping.")
            continue
        
        try:
            # Extract the binary labelmap for this segment
            segment_array = slicer.util.arrayFromSegmentBinaryLabelmap(segmentation_node, segment_id)
            unique_vals = np.unique(segment_array)
            print(f"Segment '{structure_name}' binary labelmap unique values: {unique_vals}")
            
            # Ensure the segment array matches the labelmap shape
            if segment_array.shape != labelmap_shape:
                print(f"Shape mismatch for '{structure_name}': expected {labelmap_shape}, got {segment_array.shape}. Skipping.")
                continue
            
            # Apply the custom label value where the segment is present
            mask = segment_array > 0
            final_labelmap_array[mask] = label_value
            print(f"Assigned label {label_value} to '{structure_name}' in the mask labelmap.")
            
        except Exception as e:
            print(f"Failed to process segment '{structure_name}' for mask NIfTI: {e}")
    
    # Debug: Check the unique labels in the final labelmap
    unique_labels = np.unique(final_labelmap_array)
    print(f"Unique labels in mask labelmap: {unique_labels}")
    
    # Create a new labelmap volume node for the final labelmap
    final_labelmap_volume_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    
    # Copy the geometry from the temporary labelmap node
    final_labelmap_volume_node.Copy(temp_labelmap_volume_node)
    
    # Set the array data
    slicer.util.updateVolumeFromArray(final_labelmap_volume_node, final_labelmap_array)
    
    # Save the labelmap as a NIfTI file
    success = slicer.util.saveNode(final_labelmap_volume_node, output_file_path)
    if not success:
        raise RuntimeError(f"Failed to save mask labelmap to file: {output_file_path}")
    
    print(f"Exported mask labelmap to: {output_file_path}")
    
    # Clean up temporary nodes
    slicer.mrmlScene.RemoveNode(temp_labelmap_volume_node)
    return final_labelmap_volume_node

# Function to convert NIfTI to RGB PNGs for Axial, Coronal, and Sagittal views (masks)
def nifti_to_png(nifti_path, output_dir):
    # Load NIfTI
    img = nib.load(nifti_path).get_fdata()
    print(f"Unique labels in {nifti_path}: {np.unique(img)}")
    
    # Create output directories for Axial, Coronal, and Sagittal views
    axial_dir = os.path.join(output_dir, "masks", "axial")
    coronal_dir = os.path.join(output_dir, "masks", "coronal")
    sagittal_dir = os.path.join(output_dir, "masks", "sagittal")
    for dir_path in [axial_dir, coronal_dir, sagittal_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # --- Axial View (up-down, slices along z-axis) ---
    # Axial view shows the x-y plane (left-right, back-front) at each z level
    print(f"Generating axial mask slices (total: {img.shape[2]})...")
    for z in range(img.shape[2]):
        slice = img[:, :, z]  # Shape: (x, y)
        # Initialize RGB array (height, width, 3)
        colored_slice = np.zeros((slice.shape[0], slice.shape[1], 3), dtype=np.uint8)
        for label, rgb in label_to_rgb_color.items():
            mask = slice == label
            colored_slice[mask] = rgb  # Assign RGB tuple
        # Convert to PIL Image and resize
        colored_image = Image.fromarray(colored_slice)
        colored_image = colored_image.resize((512, 512), Image.NEAREST)
        colored_image = colored_image.rotate(-90, expand=True) # Rotate 90 degrees clockwise
        colored_image.save(os.path.join(axial_dir, f'slice_{z:03d}.png'))
        print(f"Saved axial mask-only {os.path.join(axial_dir, f'slice_{z:03d}.png')}")
    
    # --- Coronal View (right-left, slices along x-axis) ---
    # Coronal view shows the y-z plane (back-front, bottom-top) at each x level
    coronal_img = np.transpose(img, (0, 1, 2))  # (x, y, z)
    print(f"Generating coronal mask slices (total: {coronal_img.shape[0]})...")
    for x in range(coronal_img.shape[0]):
        slice = coronal_img[x, :, :]  # Shape: (y, z)
        # Initialize RGB array (height, width, 3)
        colored_slice = np.zeros((slice.shape[0], slice.shape[1], 3), dtype=np.uint8)
        for label, rgb in label_to_rgb_color.items():
            mask = slice == label
            colored_slice[mask] = rgb  # Assign RGB tuple
        # Convert to PIL Image and resize
        colored_image = Image.fromarray(colored_slice)
        colored_image = colored_image.resize((512, 512), Image.NEAREST)
        colored_image = colored_image.rotate(90, expand=True) # Rotate 90 degrees counter-clockwise
        colored_image.save(os.path.join(coronal_dir, f'slice_{x:03d}.png'))
        print(f"Saved coronal mask-only {os.path.join(coronal_dir, f'slice_{x:03d}.png')}")
    
    # --- Sagittal View (front-back, slices along y-axis) ---
    # Sagittal view shows the x-z plane (left-right, bottom-top) at each y level
    sagittal_img = np.transpose(img, (1, 0, 2))  # (y, x, z)
    print(f"Generating sagittal mask slices (total: {sagittal_img.shape[0]})...")
    for y in range(sagittal_img.shape[0]):
        slice = sagittal_img[y, :, :]  # Shape: (x, z)
        # Initialize RGB array (height, width, 3)
        colored_slice = np.zeros((slice.shape[0], slice.shape[1], 3), dtype=np.uint8)
        for label, rgb in label_to_rgb_color.items():
            mask = slice == label
            colored_slice[mask] = rgb  # Assign RGB tuple
        # Convert to PIL Image and resize
        colored_image = Image.fromarray(colored_slice)
        colored_image = colored_image.resize((512, 512), Image.NEAREST)
        colored_image = colored_image.rotate(90, expand=True) # Rotate 90 degrees counter-clockwise
        colored_image.save(os.path.join(sagittal_dir, f'slice_{y:03d}.png'))
        print(f"Saved sagittal mask-only {os.path.join(sagittal_dir, f'slice_{y:03d}.png')}")
    
    print(f"Converted {nifti_path} to mask PNGs in {output_dir}")

# Function to convert NIfTI to contour PNGs for Axial, Coronal, and Sagittal views
def nifti_to_contour_png(nifti_path, output_dir, contour_thickness=1):
    """
    Convert NIfTI segmentation masks to PNGs with colored contours for Axial, Coronal, and Sagittal views.
    
    Args:
        nifti_path: Path to the NIfTI file
        output_dir: Directory to save PNG outputs
        contour_thickness: Thickness of contour lines in pixels
    """
    # Load NIfTI
    img = nib.load(nifti_path).get_fdata()
    print(f"Unique labels in {nifti_path} for contours: {np.unique(img)}")
    
    # Create output directories for Axial, Coronal, and Sagittal views
    axial_dir = os.path.join(output_dir, "contours", "axial")
    coronal_dir = os.path.join(output_dir, "contours", "coronal")
    sagittal_dir = os.path.join(output_dir, "contours", "sagittal")
    for dir_path in [axial_dir, coronal_dir, sagittal_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Get all unique label values in the image
    unique_labels = np.unique(img).astype(int)
    
    # --- Axial View (up-down, slices along z-axis) ---
    print(f"Generating axial contour slices (total: {img.shape[2]})...")
    for z in range(img.shape[2]):
        slice_data = img[:, :, z].astype(np.uint8)
        # Create a blank RGB image (black background)
        height, width = slice_data.shape
        contour_image = np.zeros((height, width, 3), dtype=np.uint8)
        # Process each label
        for label in unique_labels:
            if label == 0:  # Skip background
                continue
            # Create binary mask for this label
            binary_mask = (slice_data == label).astype(np.uint8) * 255
            # Find contours in the binary mask
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Get color for this label
            if label in label_to_rgb_color:
                # OpenCV uses BGR format, so reverse the RGB tuple
                color = label_to_rgb_color[label][::-1]
                # Draw contours on the image
                cv2.drawContours(contour_image, contours, -1, color, contour_thickness)
        # Convert from BGR (OpenCV) back to RGB (PIL)
        contour_image_rgb = cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image and resize
        contour_image_pil = Image.fromarray(contour_image_rgb)
        contour_image_pil = contour_image_pil.resize((512, 512), Image.NEAREST)
        contour_image_pil = contour_image_pil.rotate(-90, expand=True) # Rotate 90 degrees clockwise
        contour_image_pil.save(os.path.join(axial_dir, f'slice_{z:03d}.png'))
        print(f"Saved axial contour {os.path.join(axial_dir, f'slice_{z:03d}.png')}")
    
    # --- Coronal View (right-left, slices along x-axis) ---
    coronal_img = np.transpose(img, (0, 1, 2))  # (x, y, z)
    print(f"Generating coronal contour slices (total: {coronal_img.shape[0]})...")
    for x in range(coronal_img.shape[0]):
        slice_data = coronal_img[x, :, :].astype(np.uint8)  # Shape: (y, z)
        # Create a blank RGB image (black background)
        height, width = slice_data.shape
        contour_image = np.zeros((height, width, 3), dtype=np.uint8)
        # Process each label
        for label in unique_labels:
            if label == 0:  # Skip background
                continue
            # Create binary mask for this label
            binary_mask = (slice_data == label).astype(np.uint8) * 255
            # Find contours in the binary mask
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Get color for this label
            if label in label_to_rgb_color:
                # OpenCV uses BGR format, so reverse the RGB tuple
                color = label_to_rgb_color[label][::-1]
                # Draw contours on the image
                cv2.drawContours(contour_image, contours, -1, color, contour_thickness)
        # Convert from BGR (OpenCV) back to RGB (PIL)
        contour_image_rgb = cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image and resize
        contour_image_pil = Image.fromarray(contour_image_rgb)
        contour_image_pil = contour_image_pil.resize((512, 512), Image.NEAREST)
        contour_image_pil = contour_image_pil.rotate(90, expand=True) # Rotate 90 degrees counter-clockwise
        contour_image_pil.save(os.path.join(coronal_dir, f'slice_{x:03d}.png'))
        print(f"Saved coronal contour {os.path.join(coronal_dir, f'slice_{x:03d}.png')}")
    
    # --- Sagittal View (front-back, slices along y-axis) ---
    sagittal_img = np.transpose(img, (1, 0, 2))  # (y, x, z)
    print(f"Generating sagittal contour slices (total: {sagittal_img.shape[0]})...")
    for y in range(sagittal_img.shape[0]):
        slice_data = sagittal_img[y, :, :].astype(np.uint8)  # Shape: (x, z)
        # Create a blank RGB image (black background)
        height, width = slice_data.shape
        contour_image = np.zeros((height, width, 3), dtype=np.uint8)
        # Process each label
        for label in unique_labels:
            if label == 0:  # Skip background
                continue
            # Create binary mask for this label
            binary_mask = (slice_data == label).astype(np.uint8) * 255
            # Find contours in the binary mask
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Get color for this label
            if label in label_to_rgb_color:
                # OpenCV uses BGR format, so reverse the RGB tuple
                color = label_to_rgb_color[label][::-1]
                # Draw contours on the image
                cv2.drawContours(contour_image, contours, -1, color, contour_thickness)
        # Convert from BGR (OpenCV) back to RGB (PIL)
        contour_image_rgb = cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image and resize
        contour_image_pil = Image.fromarray(contour_image_rgb)
        contour_image_pil = contour_image_pil.resize((512, 512), Image.NEAREST)
        contour_image_pil = contour_image_pil.rotate(90, expand=True) # Rotate 90 degrees counter-clockwise
        contour_image_pil.save(os.path.join(sagittal_dir, f'slice_{y:03d}.png'))
        print(f"Saved sagittal contour {os.path.join(sagittal_dir, f'slice_{y:03d}.png')}")
    
    print(f"Converted {nifti_path} to contour PNGs in {output_dir}")

# Function to count unique colors across all mask PNGs
def count_unique_colors_in_masks(output_dir):
    """
    Count the unique colors across all mask PNGs in the axial, coronal, and sagittal directories.
    
    Args:
        output_dir: Directory containing the mask PNGs.
    """
    print("\nCounting unique color classes across all mask PNGs...")
    unique_colors = set()
    for subdir in ["axial", "coronal", "sagittal"]:
        dir_path = os.path.join(output_dir, "masks", subdir)
        if not os.path.exists(dir_path):
            print(f"Directory {dir_path} does not exist, skipping.")
            continue
        for file in os.listdir(dir_path):
            if file.endswith(".png"):
                img_path = os.path.join(dir_path, file)
                try:
                    img = np.array(Image.open(img_path))
                    # Convert RGB array to tuples for comparison
                    for row in img:
                        for pixel in row:
                            unique_colors.add(tuple(pixel))
                except Exception as e:
                    print(f"Failed to process {img_path}: {e}")
    
    # Print the total number of unique color classes and the colors
    print(f"Total number of unique color classes in mask PNGs: {len(unique_colors)}")
    print(f"Unique colors in mask PNGs (RGB tuples): {sorted(unique_colors)}")

# Function to sanitize a string for use as a file name
def sanitize_filename(filename):
    invalid_chars = r'<>:"/\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")
    return filename

# Main script
def export_to_nifti_and_png():
    # Get the segmentation node
    segmentation_nodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
    if not segmentation_nodes:
        raise ValueError("No segmentation nodes found in the scene.")
    segmentation_node = segmentation_nodes[0]
    print(f"Found segmentation node: {segmentation_node.GetName()}")
    
    # Get the volume node (DICOM image data)
    volume_nodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
    if not volume_nodes:
        raise ValueError("No volume nodes found in the scene.")
    volume_node = volume_nodes[0]  # Assuming the first volume node is the DICOM image data
    print(f"Found volume node: {volume_node.GetName()}")
    
    # Step 1: Check available structures and geometry
    valid_segments = check_available_structures(segmentation_node)
    
    # Step 2: Assign standardized label values
    assign_standardized_labels(segmentation_node)
    
    # Step 3: Reorder segments
    reorder_segments(segmentation_node, desired_order)
    
    # Step 4: Define output directory and file paths
    output_dir = r"file/path/to/output"
    os.makedirs(output_dir, exist_ok=True)
    sanitized_name = sanitize_filename(segmentation_node.GetName())
    mask_nifti_path = os.path.join(output_dir, f"{sanitized_name}_Refined_Nifty.nii.gz")
    
    # Step 5: Export base images from DICOM volume for Axial, Coronal, and Sagittal views
    export_base_images(volume_node, output_dir)
    
    # Step 6: Export mask NIfTI with custom labels
    mask_labelmap_volume_node = export_segmentation_as_nifti(segmentation_node, mask_nifti_path)
    
    # Step 7: Convert mask NIfTI to mask PNGs for Axial, Coronal, and Sagittal views
    nifti_to_png(mask_nifti_path, output_dir)
    
    # Step 8: Convert mask NIfTI to contour PNGs for Axial, Coronal, and Sagittal views
    contour_thickness = 1  # Set contour thickness
    nifti_to_contour_png(mask_nifti_path, output_dir, contour_thickness)
    
    # Step 9: Count unique colors in mask PNGs
    count_unique_colors_in_masks(output_dir)
    
    # Clean up
    slicer.mrmlScene.RemoveNode(mask_labelmap_volume_node)
    print("Export to NIfTI, base images, mask PNGs, and contour PNGs completed successfully.")

# Run the script
try:
    export_to_nifti_and_png()
except Exception as e:
    print(f"Error during execution: {e}")
