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

# Function to export original DICOM data as NIfTI with specified structures, labels, and order
def export_original_nifti(segmentation_node, output_file_path):
    """
    Export the segmentation as a NIfTI file with specified structures, standardized labels, and desired order.
    
    Args:
        segmentation_node: The vtkMRMLSegmentationNode containing the segmentation data.
        output_file_path: Path to save the NIfTI file.
    
    Returns:
        The labelmap volume node created during the process (for cleanup purposes).
    """
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
    print(f"Original labelmap shape: {labelmap_shape}")
    
    # Initialize the final labelmap array with zeros (background)
    final_labelmap_array = np.zeros(labelmap_shape, dtype=np.uint8)
    
    # Iterate over segments in the desired order and assign standardized labels
    print("\nConstructing original NIfTI with specified structures, labels, and order:")
    for segment_name in desired_order:
        if segment_name not in segment_name_to_id:
            print(f"Segment '{segment_name}' not found, skipping.")
            continue
        if segment_name not in included_structures:
            print(f"Segment '{segment_name}' not in included_structures, skipping.")
            continue
        
        segment_id = segment_name_to_id[segment_name]
        label_value = standard_label_mapping.get(segment_name)
        if label_value is None:
            print(f"No label value defined for '{segment_name}', skipping.")
            continue
        
        try:
            # Extract the binary labelmap for this segment
            segment_array = slicer.util.arrayFromSegmentBinaryLabelmap(segmentation_node, segment_id)
            unique_vals = np.unique(segment_array)
            print(f"Segment '{segment_name}' binary labelmap unique values: {unique_vals}")
            
            # Ensure the segment array matches the labelmap shape
            if segment_array.shape != labelmap_shape:
                print(f"Shape mismatch for '{segment_name}': expected {labelmap_shape}, got {segment_array.shape}. Skipping.")
                continue
            
            # Apply the standardized label value where the segment is present
            mask = segment_array > 0
            final_labelmap_array[mask] = label_value
            print(f"Assigned label {label_value} to '{segment_name}' in the original NIfTI.")
            
        except Exception as e:
            print(f"Failed to process segment '{segment_name}' for original NIfTI: {e}")
    
    # Debug: Check the unique labels in the final labelmap
    unique_labels = np.unique(final_labelmap_array)
    print(f"Unique labels in final original NIfTI: {unique_labels}")
    
    # Create a new labelmap volume node for the final labelmap
    final_labelmap_volume_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    
    # Copy the geometry from the temporary labelmap node
    final_labelmap_volume_node.Copy(temp_labelmap_volume_node)
    
    # Set the array data
    slicer.util.updateVolumeFromArray(final_labelmap_volume_node, final_labelmap_array)
    
    # Save the labelmap as a NIfTI file
    success = slicer.util.saveNode(final_labelmap_volume_node, output_file_path)
    if not success:
        raise RuntimeError(f"Failed to save original NIfTI to file: {output_file_path}")
    
    print(f"Exported original NIfTI to: {output_file_path}")
    
    # Clean up temporary nodes
    slicer.mrmlScene.RemoveNode(temp_labelmap_volume_node)
    return final_labelmap_volume_node

# Function to sanitize a string for use as a file name
def sanitize_filename(filename):
    invalid_chars = r'<>:"/\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")
    return filename

# Main script
def export_to_nifti():
    # Get the segmentation node
    segmentation_nodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
    if not segmentation_nodes:
        raise ValueError("No segmentation nodes found in the scene.")
    segmentation_node = segmentation_nodes[0]
    print(f"Found segmentation node: {segmentation_node.GetName()}")
    
    # Step 1: Check available structures and geometry
    valid_segments = check_available_structures(segmentation_node)
    
    # Step 2: Assign standardized label values
    assign_standardized_labels(segmentation_node)
    
    # Step 3: Reorder segments
    reorder_segments(segmentation_node, desired_order)
    
    # Step 4: Define output directory and file path
    output_dir = r"file/path/to/output"
    os.makedirs(output_dir, exist_ok=True)
    sanitized_name = sanitize_filename(segmentation_node.GetName())
    original_nifti_path = os.path.join(output_dir, f"{sanitized_name}_Original_nifty.nii.gz")
    
    # Step 5: Export original NIfTI with specified structures, labels, and order
    original_labelmap_volume_node = export_original_nifti(segmentation_node, original_nifti_path)
    
    # Clean up
    slicer.mrmlScene.RemoveNode(original_labelmap_volume_node)
    print("Export to original NIfTI completed successfully.")

# Run the script
try:
    export_to_nifti()
except Exception as e:
    print(f"Error during execution: {e}")
