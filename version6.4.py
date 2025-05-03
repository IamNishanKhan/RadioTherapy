import slicer
import os
import numpy as np
from PIL import Image
import nibabel as nib
import cv2
import qt

# Setting Path for the output directory
# This is the directory where the output files will be saved
folder_to_output = r"file/path/to/output"
# Make sure to change this to your desired output directory
# Note: The path should be a valid directory on your system

# Function to extract patient ID from DICOM database
def extract_patient_id():
    # Step 1: Get the segmentation and volume nodes
    segmentation_nodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
    if not segmentation_nodes:
        raise ValueError("No segmentation nodes found in the scene.")
    segmentation_node = segmentation_nodes[0]
    print(f"Found segmentation node: {segmentation_node.GetName()}")

    volume_nodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
    if not volume_nodes:
        raise ValueError("No volume nodes found in the scene.")
    volume_node = volume_nodes[0]
    print(f"Found volume node: {volume_node.GetName()}")

    # Step 2: Use the subject hierarchy to find the study/series associated with the segmentation node
    shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
    if not shNode:
        print("Subject hierarchy node not found.")
    else:
        # Get the subject hierarchy item for the segmentation node
        seg_item_id = shNode.GetItemByDataNode(segmentation_node)
        if seg_item_id:
            # Traverse up the hierarchy to find the study node
            current_item_id = seg_item_id
            study_item_id = None
            while current_item_id:
                item_level = shNode.GetItemLevel(current_item_id)
                if item_level == "Study":
                    study_item_id = current_item_id
                    break
                current_item_id = shNode.GetItemParent(current_item_id)

            if study_item_id:
                study_uid = shNode.GetItemUID(study_item_id, "DICOM")
                print(f"Found study UID in subject hierarchy: {study_uid}")
            else:
                print("No study node found in subject hierarchy for the segmentation node.")
                study_uid = None
        else:
            print("Segmentation node not found in subject hierarchy.")
            study_uid = None

    # Step 3: Extract patient ID from DICOM database by matching the study UID
    try:
        dicom_database = slicer.dicomDatabase
        if not dicom_database:
            raise ValueError("DICOM database not available in Slicer.")

        # Get all patients in the DICOM database
        patients = dicom_database.patients()
        if not patients:
            raise ValueError("No patients found in DICOM database.")

        # If we found a study UID, find the patient associated with that study
        if study_uid:
            patient_uid = None
            for patient in patients:
                studies = dicom_database.studiesForPatient(patient)
                if study_uid in studies:
                    patient_uid = patient
                    break

            if patient_uid:
                # Get a DICOM file for this patient to extract the PatientID
                studies = dicom_database.studiesForPatient(patient_uid)
                series = dicom_database.seriesForStudy(studies[0])
                if not series:
                    raise ValueError(f"No series found for study UID: {studies[0]}")
                files = dicom_database.filesForSeries(series[0])
                if not files:
                    raise ValueError(f"No files found for series UID: {series[0]}")
                dicom_file = files[0]
                patient_id = dicom_database.fileValue(dicom_file, "0010,0020")  # PatientID tag
                if patient_id:
                    print(f"Extracted patient ID from DICOM database for study UID {study_uid}: {patient_id}")
                    return patient_id
                else:
                    raise ValueError(f"PatientID not found in DICOM database for patient UID: {patient_uid}")
            else:
                raise ValueError(f"No patient found in DICOM database for study UID: {study_uid}")
        else:
            # If no study UID was found, fall back to checking all patients
            if len(patients) == 1:
                # If there's only one patient, use that
                patient_uid = patients[0]
                studies = dicom_database.studiesForPatient(patient_uid)
                if not studies:
                    raise ValueError(f"No studies found for patient UID: {patient_uid}")
                series = dicom_database.seriesForStudy(studies[0])
                if not series:
                    raise ValueError(f"No series found for study UID: {studies[0]}")
                files = dicom_database.filesForSeries(series[0])
                if not files:
                    raise ValueError(f"No files found for series UID: {series[0]}")
                dicom_file = files[0]
                patient_id = dicom_database.fileValue(dicom_file, "0010,0020")  # PatientID tag
                if patient_id:
                    print(f"Extracted patient ID from DICOM database (single patient): {patient_id}")
                    return patient_id
                else:
                    raise ValueError(f"PatientID not found in DICOM database for patient UID: {patient_uid}")
            else:
                # Multiple patients found, and we couldn't match the study UID
                raise ValueError("Multiple patients found in DICOM database, and study UID could not be matched.")

    except Exception as e:
        print(f"Failed to extract patient ID from DICOM database: {e}")

    # Step 4: Fallback to user input if DICOM database fails
    print("Could not extract patient ID automatically.")
    patient_id = qt.QInputDialog.getText(None, "Enter Patient ID", "Please enter the patient ID (e.g., R1807009876):")
    if patient_id:
        print(f"User provided patient ID: {patient_id}")
        return patient_id
    else:
        raise ValueError("No patient ID provided by user.")


# Attempt to extract patient ID from DICOM database
# This will be used to create the output directory structure
try:
    patient_id = extract_patient_id()
except Exception as e:
    print(f"Error extracting patient ID: {e}")


def export_base_hdr_images():
    """
    Export axial, coronal, and sagittal HDR-like PNG images from the currently loaded volume in Slicer.
    
    Parameters:
    - patient_id (str): The patient ID to use in the output directory structure.
    - base_output_dir (str): The base directory where the output will be saved.
    
    Output:
    - Saves PNGs in {base_output_dir}/{patient_id}_ExtractedData/{patient_id}_base_hdr/{patient_id}_base_hdr_{axial/coronal/sagittal}/
    """
    
    # Set the base output directory
    base_output_dir = folder_to_output  # Use the directory from the dialog
    
    # Define the output directory structure
    output_dir = os.path.join(base_output_dir, f"{patient_id}_ExtractedData")
    
    # Create output directories for Axial, Coronal, and Sagittal base HDR PNGs with patient ID
    axial_dir = os.path.join(output_dir, f"{patient_id}_base_hdr", f"{patient_id}_axial_base_hdr")
    coronal_dir = os.path.join(output_dir, f"{patient_id}_base_hdr", f"{patient_id}_coronal_base_hdr")
    sagittal_dir = os.path.join(output_dir, f"{patient_id}_base_hdr", f"{patient_id}_sagittal_base_hdr")
    
    for dir_path in [axial_dir, coronal_dir, sagittal_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Get the currently loaded volume node in Slicer
    volume_nodes = slicer.util.getNodes('vtkMRMLScalarVolumeNode*')
    if not volume_nodes:
        raise ValueError("No volume nodes found in the scene. Please load a DICOM volume first.")

    # Take the first volume node (modify if you need a specific node)
    volume_node = list(volume_nodes.values())[0]
    print(f"Using volume node: {volume_node.GetName()}")

    # Extract the image data as a NumPy array
    # Shape will be (z, y, x) for axial slices
    image_volume = slicer.util.arrayFromVolume(volume_node)
    print(f"Image volume shape: {image_volume.shape}")

    # Determine pixel representation (assuming 16-bit data, common in DICOM)
    # In Slicer, the array is typically float or int16; we'll treat it as int16 for consistency
    image_volume = image_volume.astype(np.int16)

    # Debugging: Print original pixel range
    print(f"Original pixel range: {np.min(image_volume)} to {np.max(image_volume)}")

    # Windowing parameters (same as original script)
    window_center = 50  # Adjusted for soft tissue (e.g., pelvis)
    window_width = 400  # Wider range to capture more detail

    # Calculate lower and upper bounds for windowing
    lower_bound = window_center - window_width / 2
    upper_bound = window_center + window_width / 2

    # Apply windowing: clip pixel values outside the window range
    image_volume = np.clip(image_volume, lower_bound, upper_bound)

    # Debugging: Print pixel range after windowing
    print(f"Pixel range after windowing: {np.min(image_volume)} to {np.max(image_volume)}")

    # Normalize the pixel values to the range [0, 255]
    if upper_bound > lower_bound:
        image_volume = ((image_volume - lower_bound) / (upper_bound - lower_bound) * 255).astype(np.uint8)
    else:
        # Handle edge case where all pixel values are the same
        image_volume = np.zeros_like(image_volume, dtype=np.uint8)

    # Generate PNGs for all three views
    # Axial view: slices along z-axis (x, y planes)
    print(f"Generating axial images (total: {image_volume.shape[0]})...")
    for z in range(image_volume.shape[0]):
        slice_data = image_volume[z, :, :]  # Shape: (y, x)
        # Convert grayscale to RGB by replicating the channel
        rgb_data = np.stack([slice_data] * 3, axis=-1)  # Shape: (y, x, 3)
        image = Image.fromarray(rgb_data.astype(np.uint8), mode='RGB')
        image = image.resize((512, 512), Image.LANCZOS)  # Use LANCZOS for smoother resizing
        output_path = os.path.join(axial_dir, f"{patient_id}_{z:03d}.png")
        image.save(output_path)

    # Coronal view: slices along x-axis (y, z planes)
    coronal_volume = image_volume  # Already in (z, y, x)
    print(f"Generating coronal images (total: {coronal_volume.shape[2]})...")
    for x in range(coronal_volume.shape[2]):  # Iterate over x (left to right)
        slice_data = coronal_volume[:, :, x]  # Shape: (z, y)
        # Convert grayscale to RGB by replicating the channel
        rgb_data = np.stack([slice_data] * 3, axis=-1)  # Shape: (z, y, 3)
        image = Image.fromarray(rgb_data.astype(np.uint8), mode='RGB')
        image = image.resize((512, 512), Image.LANCZOS)  # Use LANCZOS for smoother resizing
        image = image.transpose(Image.FLIP_TOP_BOTTOM)  # Flip vertically
        output_path = os.path.join(coronal_dir, f"{patient_id}_{x:03d}.png")
        image.save(output_path)

    # Sagittal view: slices along y-axis (x, z planes)
    sagittal_volume = np.transpose(image_volume, (1, 0, 2))  # (z, y, x) -> (y, z, x)
    print(f"Generating sagittal images (total: {sagittal_volume.shape[0]})...")
    for y in range(sagittal_volume.shape[0]):  # Iterate over y (posterior to anterior)
        slice_data = sagittal_volume[y, :, :]  # Shape: (z, x)
        # Convert grayscale to RGB by replicating the channel
        rgb_data = np.stack([slice_data] * 3, axis=-1)  # Shape: (z, x, 3)
        image = Image.fromarray(rgb_data.astype(np.uint8), mode='RGB')
        image = image.resize((512, 512), Image.LANCZOS)  # Use LANCZOS for smoother resizing
        image = image.transpose(Image.FLIP_TOP_BOTTOM)  # Flip vertically
        output_path = os.path.join(sagittal_dir, f"{patient_id}_{y:03d}.png")
        image.save(output_path)

    print("Saved axial, coronal, and sagittal PNGs completed.")


# Run the export with try-except
try:
    export_base_hdr_images()
except Exception as e:
    print(f"Error during original images export: {e}")
    
    
# Function to export the original volume (images) as a NIfTI file
def export_original_images_as_nifti(volume_node, output_file_path):
    # Get the image data from the volume node
    image_data = slicer.util.arrayFromVolume(volume_node)
    print(f"Original image data shape: {image_data.shape}")

    # Create a new volume node for export (optional, but ensures clean export)
    export_volume_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    export_volume_node.Copy(volume_node)  # Copy the original volume node to preserve metadata
    slicer.util.updateVolumeFromArray(export_volume_node, image_data)

    # Save the volume as a NIfTI file
    success = slicer.util.saveNode(export_volume_node, output_file_path)
    if not success:
        raise RuntimeError(f"Failed to save original images to NIfTI file: {output_file_path}")

    print(f"Exported original images to NIfTI: {output_file_path}")

    # Clean up the temporary node
    slicer.mrmlScene.RemoveNode(export_volume_node)

# Main script to run the export
def export_original_images_nifti():
    # Get the volume node (original DICOM image data)
    volume_nodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
    if not volume_nodes:
        raise ValueError("No volume nodes found in the scene.")
    volume_node = volume_nodes[0]  # Assuming the first volume node is the original image data
    print(f"Found volume node: {volume_node.GetName()}")
    
    # Define output path for the original images NIfTI file
    base_output_dir = folder_to_output  # Use the directory from the dialog
    output_dir = os.path.join(base_output_dir, f"{patient_id}_ExtractedData")
    os.makedirs(output_dir, exist_ok=True)
    original_nifti_path = os.path.join(output_dir, f"{patient_id}_Base_Nifty.nii.gz")

    # Export the original images as NIfTI
    export_original_images_as_nifti(volume_node, original_nifti_path)

# Run the export
try:
    export_original_images_nifti()
except Exception as e:
    print(f"Error during original images NIfTI export: {e}")


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
def export_base_images(volume_node, output_dir, patient_id):
    # Create output directories for Axial, Coronal, and Sagittal views with patient ID
    axial_dir = os.path.join(output_dir, f"{patient_id}_base", f"{patient_id}_axial_base")
    coronal_dir = os.path.join(output_dir, f"{patient_id}_base", f"{patient_id}_coronal_base")
    sagittal_dir = os.path.join(output_dir, f"{patient_id}_base", f"{patient_id}_sagittal_base")
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
    print(f"Generating coronal base images (total: {image_data.shape[2]})...")
    for z in range(image_data.shape[2]):
        slice = image_data[:, :, z]  # Shape: (x, y)
        # Convert grayscale to RGB by replicating the channel
        rgb_data = np.stack([slice] * 3, axis=-1)  # Shape: (x, y, 3)
        # Convert to PIL Image and resize
        base_image = Image.fromarray(rgb_data.astype(np.uint8), mode='RGB')
        base_image = base_image.resize((512, 512), Image.NEAREST)
        base_image = base_image.transpose(Image.FLIP_TOP_BOTTOM)  # Flip vertically
        base_image.save(os.path.join(coronal_dir, f'{patient_id}_{z:03d}.png'))

    # --- Axial View (right-left, slices along x-axis) ---
    # Axial view shows the y-z plane (back-front, bottom-top) at each x level
    axial_img = np.transpose(image_data, (0, 1, 2))  # (x, y, z)
    print(f"Generating axial base images (total: {axial_img.shape[0]})...")
    for x in range(axial_img.shape[0]):
        slice = axial_img[x, :, :]  # Shape: (y, z)
        # Convert grayscale to RGB by replicating the channel
        rgb_data = np.stack([slice] * 3, axis=-1)  # Shape: (y, z, 3)
        # Convert to PIL Image and resize
        base_image = Image.fromarray(rgb_data.astype(np.uint8), mode='RGB')
        base_image = base_image.resize((512, 512), Image.NEAREST)
        base_image.save(os.path.join(axial_dir, f'{patient_id}_{x:03d}.png'))

    # --- Sagittal View (front-back, slices along y-axis) ---
    # Sagittal view shows the x-z plane (left-right, bottom-top) at each y level
    sagittal_img = np.transpose(image_data, (1, 0, 2))  # (y, x, z)
    print(f"Generating sagittal base images (total: {sagittal_img.shape[0]})...")
    for y in range(sagittal_img.shape[0]):
        slice = sagittal_img[y, :, :]  # Shape: (x, z)
        # Convert grayscale to RGB by replicating the channel
        rgb_data = np.stack([slice] * 3, axis=-1)  # Shape: (x, z, 3)
        # Convert to PIL Image and resize
        base_image = Image.fromarray(rgb_data.astype(np.uint8), mode='RGB')
        base_image = base_image.resize((512, 512), Image.NEAREST)
        base_image = base_image.transpose(Image.FLIP_TOP_BOTTOM)  # Flip vertically
        base_image.save(os.path.join(sagittal_dir, f'{patient_id}_{y:03d}.png'))
    
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
def nifti_to_png(nifti_path, output_dir, patient_id):
    # Load NIfTI
    img = nib.load(nifti_path).get_fdata()
    print(f"Unique labels in {nifti_path}: {np.unique(img)}")
    
    # Create output directories for Axial, Coronal, and Sagittal views with patient ID
    axial_dir = os.path.join(output_dir, f"{patient_id}_mask", f"{patient_id}_axial_mask")
    coronal_dir = os.path.join(output_dir, f"{patient_id}_mask", f"{patient_id}_coronal_mask")
    sagittal_dir = os.path.join(output_dir, f"{patient_id}_mask", f"{patient_id}_sagittal_mask")
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
        colored_image.save(os.path.join(axial_dir, f'{patient_id}_{z:03d}.png'))
    
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
        colored_image.save(os.path.join(coronal_dir, f'{patient_id}_{x:03d}.png'))
    
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
        colored_image.save(os.path.join(sagittal_dir, f'{patient_id}_{y:03d}.png'))
    
    print(f"Converted {nifti_path} to mask PNGs in {output_dir}")

# Function to convert NIfTI to contour PNGs for Axial, Coronal, and Sagittal views
def nifti_to_contour_png(nifti_path, output_dir, patient_id, contour_thickness=1):
    """
    Convert NIfTI segmentation masks to PNGs with colored contours for Axial, Coronal, and Sagittal views.
    
    Args:
        nifti_path: Path to the NIfTI file
        output_dir: Directory to save PNG outputs
        patient_id: Patient ID to include in filenames
        contour_thickness: Thickness of contour lines in pixels
    """
    # Load NIfTI
    img = nib.load(nifti_path).get_fdata()
    print(f"Unique labels in {nifti_path} for contours: {np.unique(img)}")
    
    # Create output directories for Axial, Coronal, and Sagittal views with patient ID
    axial_dir = os.path.join(output_dir, f"{patient_id}_contour", f"{patient_id}_axial_contour")
    coronal_dir = os.path.join(output_dir, f"{patient_id}_contour", f"{patient_id}_coronal_contour")
    sagittal_dir = os.path.join(output_dir, f"{patient_id}_contour", f"{patient_id}_sagittal_contour")
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
        contour_image_pil.save(os.path.join(axial_dir, f'{patient_id}_{z:03d}.png'))
    
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
        contour_image_pil.save(os.path.join(coronal_dir, f'{patient_id}_{x:03d}.png'))
    
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
        contour_image_pil.save(os.path.join(sagittal_dir, f'{patient_id}_{y:03d}.png'))
    
    print(f"Converted {nifti_path} to contour PNGs in {output_dir}")

# Function to count unique colors across all mask PNGs
def count_unique_colors_in_masks(output_dir, patient_id):
    """
    Count the unique colors across all mask PNGs in the axial, coronal, and sagittal directories.
    
    Args:
        output_dir: Directory containing the mask PNGs.
        patient_id: Patient ID to include in directory paths
    """
    print("\nCounting unique color classes across all mask PNGs...")
    unique_colors = set()
    for subdir in [f"{patient_id}_axial_mask", f"{patient_id}_coronal_mask", f"{patient_id}_sagittal_mask"]:
        dir_path = os.path.join(output_dir, f"{patient_id}_mask", subdir)
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
    
    # Step 4: Define output directory and file paths with patient ID
    base_output_dir = folder_to_output  # Change this to your desired output directory
    output_dir = os.path.join(base_output_dir, f"{patient_id}_ExtractedData")
    os.makedirs(output_dir, exist_ok=True)
    mask_nifti_path = os.path.join(output_dir, f"{patient_id}_Labeled_Nifty.nii.gz")
    
    # Step 5: Export base images from DICOM volume for Axial, Coronal, and Sagittal views
    export_base_images(volume_node, output_dir, patient_id)
    
    # Step 6: Export mask NIfTI with custom labels
    mask_labelmap_volume_node = export_segmentation_as_nifti(segmentation_node, mask_nifti_path)
    
    # Step 7: Convert mask NIfTI to mask PNGs for Axial, Coronal, and Sagittal views
    nifti_to_png(mask_nifti_path, output_dir, patient_id)
    
    # Step 8: Convert mask NIfTI to contour PNGs for Axial, Coronal, and Sagittal views
    contour_thickness = 1  # Set contour thickness
    nifti_to_contour_png(mask_nifti_path, output_dir, patient_id, contour_thickness)
    
    # Step 9: Count unique colors in mask PNGs
    # count_unique_colors_in_masks(output_dir, patient_id)
    
    # Clean up
    slicer.mrmlScene.RemoveNode(mask_labelmap_volume_node)
    print("Export to NIfTI, base images, mask PNGs, and contour PNGs completed successfully.")

# Run the script
try:
    export_to_nifti_and_png()
except Exception as e:
    print(f"Error during execution: {e}")
