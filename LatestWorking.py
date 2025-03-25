import slicer
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Define the structures to include (excluding BODY)
included_structures = [
    "URINARY BLADDER",
    "SMALL BOWEL",
    "RECTUM",
    "FEMORAL HEAD_RT",
    "FEMORAL HEAD_LT",
    "GTV",
    "CTV"
]

# Define the desired order and colors
desired_order = [
    "CTV",
    "GTV",
    "FEMORAL HEAD_RT",
    "FEMORAL HEAD_LT",
    "SMALL BOWEL",
    "URINARY BLADDER",
    "RECTUM"  # Top-most (drawn last)
]

structure_colors = {
    "URINARY BLADDER": "#00ffff",  # Cyan
    "SMALL BOWEL": "#ff9299",      # Light Pink
    "RECTUM": "#894040",           # Brown
    "FEMORAL HEAD_RT": "#ffff00",  # Yellow
    "FEMORAL HEAD_LT": "#ffff00",  # Yellow
    "GTV": "#ff3cff",              # Magenta-Pink
    "CTV": "#3737ff"               # Blue
}

# Define the standardized label mapping (excluding BODY)
standard_label_mapping = {
    "URINARY BLADDER": 2,
    "SMALL BOWEL": 3,
    "RECTUM": 4,
    "FEMORAL HEAD_RT": 5,
    "FEMORAL HEAD_LT": 6,
    "GTV": 7,
    "CTV": 8,
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

# Function to convert segmentation to labelmap and export as NIfTI
def export_segmentation_as_nifti(segmentation_node, output_file_path):
    segmentation = segmentation_node.GetSegmentation()
    segment_ids = segmentation.GetSegmentIDs()
    
    # Debug: Check individual segment arrays before export
    print("\nChecking individual segment arrays before export:")
    for segment_id in segment_ids:
        segment_name = segmentation.GetSegment(segment_id).GetName()
        assigned_label = segmentation.GetSegment(segment_id).GetLabelValue()
        try:
            array = slicer.util.arrayFromSegmentBinaryLabelmap(segmentation_node, segment_id)
            unique_vals = np.unique(array)
            print(f"Segment '{segment_name}' (label {assigned_label}) unique values: {unique_vals}")
        except Exception as e:
            print(f"Failed to extract array for segment '{segment_name}': {e}")
    
    # Force binary labelmap regeneration
    print("\nForcing binary labelmap regeneration...")
    segmentation.CreateRepresentation("Binary labelmap", True)
    
    # Create and export labelmap
    labelmap_volume_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    success = slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
        segmentation_node, labelmap_volume_node, slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY
    )
    if not success:
        raise RuntimeError("Failed to convert segmentation to labelmap.")
    
    print("Successfully converted segmentation to labelmap.")
    
    # Debug: Inspect the final labelmap data
    labelmap_array = slicer.util.arrayFromVolume(labelmap_volume_node)
    unique_labels = np.unique(labelmap_array)
    print(f"Unique labels in final labelmap before saving: {unique_labels}")
    
    # Save the labelmap as a NIfTI file
    success = slicer.util.saveNode(labelmap_volume_node, output_file_path)
    if not success:
        raise RuntimeError(f"Failed to save labelmap to file: {output_file_path}")
    
    print(f"Exported labelmap to: {output_file_path}")
    return labelmap_volume_node

# Function to convert NIfTI labelmap to PNGs (masks only and base images without masks)
def convert_labelmap_to_png(labelmap_volume_node, output_dir, volume_node):
    # Create output directories for masks only
    axial_dir = os.path.join(output_dir, "pngs", "axial")
    coronal_dir = os.path.join(output_dir, "pngs", "coronal")
    sagittal_dir = os.path.join(output_dir, "pngs", "sagittal")
    # Create output directories for base images without masks
    base_axial_dir = os.path.join(output_dir, "base_pngs", "axial")
    base_coronal_dir = os.path.join(output_dir, "base_pngs", "coronal")
    base_sagittal_dir = os.path.join(output_dir, "base_pngs", "sagittal")
    for dir_path in [axial_dir, coronal_dir, sagittal_dir, base_axial_dir, base_coronal_dir, base_sagittal_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Get the labelmap array
    labelmap_array = slicer.util.arrayFromVolume(labelmap_volume_node)
    dims = labelmap_array.shape  # (z, y, x)
    
    # Get the image data and normalize it
    image_array = slicer.util.arrayFromVolume(volume_node)
    image_array = image_array.astype(float)
    image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array)) * 255
    image_array = image_array.astype(np.uint8)
    
    # --- Axial View ---
    print(f"Generating axial slices (total: {dims[0]})...")
    for z in range(dims[0]):
        label_slice = labelmap_array[z, :, :]
        img_slice = image_array[z, :, :]
        
        # Masks only
        fig, ax = plt.subplots()
        ax.set_facecolor('black')  # Set background to black
        
        has_contours = False
        for structure_name in desired_order:
            label_value = standard_label_mapping.get(structure_name)
            if label_value is None:
                continue
            mask = (label_slice == label_value)
            if np.any(mask):
                has_contours = True
                contours = plt.contour(mask, levels=[0.5], colors=[structure_colors[structure_name]])
                for contour in contours.collections:
                    for path in contour.get_paths():
                        poly = Polygon(
                            path.vertices,
                            fill=True,
                            facecolor=structure_colors[structure_name],
                            alpha=0.5,
                            edgecolor=structure_colors[structure_name],
                            linewidth=1
                        )
                        ax.add_patch(poly)
        
        ax.axis('off')
        temp_file = os.path.join(axial_dir, f'temp_slice_{z:03d}.png')
        plt.savefig(temp_file, bbox_inches='tight', pad_inches=0, dpi=100, facecolor='black')
        plt.close()
        
        img = Image.open(temp_file)
        img_resized = img.resize((512, 512), Image.NEAREST)
        final_output_path = os.path.join(axial_dir, f'slice_{z:03d}.png')
        img_resized.save(final_output_path)
        print(f"Saved axial mask-only {final_output_path} (Contours present: {has_contours})")
        os.remove(temp_file)
        
        # Base image without masks
        fig, ax = plt.subplots()
        ax.imshow(img_slice, cmap='gray')
        ax.axis('off')
        temp_file = os.path.join(base_axial_dir, f'temp_slice_{z:03d}.png')
        plt.savefig(temp_file, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()
        
        img = Image.open(temp_file)
        img_resized = img.resize((512, 512), Image.NEAREST)
        final_output_path = os.path.join(base_axial_dir, f'slice_{z:03d}.png')
        img_resized.save(final_output_path)
        print(f"Saved axial base image (no masks) {final_output_path}")
        os.remove(temp_file)
    
    # --- Coronal View ---
    print(f"Generating coronal slices (total: {dims[2]})...")
    coronal_label = np.transpose(labelmap_array, (2, 0, 1))  # (x, z, y)
    coronal_image = np.transpose(image_array, (2, 0, 1))
    for x in range(dims[2]):
        label_slice = coronal_label[x, :, :]
        img_slice = coronal_image[x, :, :]
        
        # Masks only
        fig, ax = plt.subplots()
        ax.set_facecolor('black')
        
        has_contours = False
        for structure_name in desired_order:
            label_value = standard_label_mapping.get(structure_name)
            if label_value is None:
                continue
            mask = (label_slice == label_value)
            if np.any(mask):
                has_contours = True
                contours = plt.contour(mask, levels=[0.5], colors=[structure_colors[structure_name]])
                for contour in contours.collections:
                    for path in contour.get_paths():
                        poly = Polygon(
                            path.vertices,
                            fill=True,
                            facecolor=structure_colors[structure_name],
                            alpha=0.5,
                            edgecolor=structure_colors[structure_name],
                            linewidth=1
                        )
                        ax.add_patch(poly)
        
        ax.axis('off')
        temp_file = os.path.join(coronal_dir, f'temp_slice_{x:03d}.png')
        plt.savefig(temp_file, bbox_inches='tight', pad_inches=0, dpi=100, facecolor='black')
        plt.close()
        
        img = Image.open(temp_file)
        img_resized = img.resize((512, 512), Image.NEAREST)
        final_output_path = os.path.join(coronal_dir, f'slice_{x:03d}.png')
        img_resized.save(final_output_path)
        print(f"Saved coronal mask-only {final_output_path} (Contours present: {has_contours})")
        os.remove(temp_file)
        
        # Base image without masks
        fig, ax = plt.subplots()
        ax.imshow(img_slice, cmap='gray')
        ax.axis('off')
        temp_file = os.path.join(base_coronal_dir, f'temp_slice_{x:03d}.png')
        plt.savefig(temp_file, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()
        
        img = Image.open(temp_file)
        img_resized = img.resize((512, 512), Image.NEAREST)
        final_output_path = os.path.join(base_coronal_dir, f'slice_{x:03d}.png')
        img_resized.save(final_output_path)
        print(f"Saved coronal base image (no masks) {final_output_path}")
        os.remove(temp_file)
    
    # --- Sagittal View ---
    print(f"Generating sagittal slices (total: {dims[1]})...")
    sagittal_label = np.transpose(labelmap_array, (1, 0, 2))  # (y, z, x)
    sagittal_image = np.transpose(image_array, (1, 0, 2))
    for y in range(dims[1]):
        label_slice = sagittal_label[y, :, :]
        img_slice = sagittal_image[y, :, :]
        
        # Masks only
        fig, ax = plt.subplots()
        ax.set_facecolor('black')
        
        has_contours = False
        for structure_name in desired_order:
            label_value = standard_label_mapping.get(structure_name)
            if label_value is None:
                continue
            mask = (label_slice == label_value)
            if np.any(mask):
                has_contours = True
                contours = plt.contour(mask, levels=[0.5], colors=[structure_colors[structure_name]])
                for contour in contours.collections:
                    for path in contour.get_paths():
                        poly = Polygon(
                            path.vertices,
                            fill=True,
                            facecolor=structure_colors[structure_name],
                            alpha=0.5,
                            edgecolor=structure_colors[structure_name],
                            linewidth=1
                        )
                        ax.add_patch(poly)
        
        ax.axis('off')
        temp_file = os.path.join(sagittal_dir, f'temp_slice_{y:03d}.png')
        plt.savefig(temp_file, bbox_inches='tight', pad_inches=0, dpi=100, facecolor='black')
        plt.close()
        
        img = Image.open(temp_file)
        img_resized = img.resize((512, 512), Image.NEAREST)
        final_output_path = os.path.join(sagittal_dir, f'slice_{y:03d}.png')
        img_resized.save(final_output_path)
        print(f"Saved sagittal mask-only {final_output_path} (Contours present: {has_contours})")
        os.remove(temp_file)
        
        # Base image without masks
        fig, ax = plt.subplots()
        ax.imshow(img_slice, cmap='gray')
        ax.axis('off')
        temp_file = os.path.join(base_sagittal_dir, f'temp_slice_{y:03d}.png')
        plt.savefig(temp_file, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()
        
        img = Image.open(temp_file)
        img_resized = img.resize((512, 512), Image.NEAREST)
        final_output_path = os.path.join(base_sagittal_dir, f'slice_{y:03d}.png')
        img_resized.save(final_output_path)
        print(f"Saved sagittal base image (no masks) {final_output_path}")
        os.remove(temp_file)

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
    
    # Get the volume node (CT/MR image)
    volume_nodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
    if not volume_nodes:
        raise ValueError("No volume nodes found in the scene.")
    volume_node = volume_nodes[0]
    print(f"Found volume node: {volume_node.GetName()}")
    
    # Step 1: Check available structures and geometry
    valid_segments = check_available_structures(segmentation_node)
    
    # Step 2: Assign standardized label values
    assign_standardized_labels(segmentation_node)
    
    # Step 3: Reorder segments
    reorder_segments(segmentation_node, desired_order)
    
    # Step 4: Define output directory and file path for NIfTI
    output_dir = r"D:\Research\Radiotherapy\Latest"
    os.makedirs(output_dir, exist_ok=True)
    sanitized_name = sanitize_filename(segmentation_node.GetName())
    nifti_file_path = os.path.join(output_dir, f"{sanitized_name}_masks_only.nii.gz")
    
    # Step 5: Export to NIfTI
    labelmap_volume_node = export_segmentation_as_nifti(segmentation_node, nifti_file_path)
    
    # Step 6: Convert NIfTI to PNGs (masks only and base images without masks)
    convert_labelmap_to_png(labelmap_volume_node, output_dir, volume_node)
    
    # Clean up
    slicer.mrmlScene.RemoveNode(labelmap_volume_node)
    print("Export to NIfTI and PNGs completed successfully.")

# Run the script
try:
    export_to_nifti_and_png()
except Exception as e:
    print(f"Error during execution: {e}")
