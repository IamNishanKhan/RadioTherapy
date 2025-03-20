import os
import slicer

# Set the output directory
output_dir = r"C:\Users\USER\Desktop\Work\Old data\2018\test"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get the current layouts and views
layoutManager = slicer.app.layoutManager()

# Find the loaded volume node - using a more robust approach
def get_volume_node():
    # Get all volume nodes
    volume_nodes = slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode')
    # First check for nodes with "CT" in the name
    for node in volume_nodes:
        if "CT" in node.GetName():
            return node
    # If no CT node found, return the first volume node if any exist
    if volume_nodes:
        return volume_nodes[0]
    return None

# Find the segmentation node
def get_segmentation_node():
    # Get all segmentation nodes
    seg_nodes = slicer.util.getNodesByClass('vtkMRMLSegmentationNode')
    if seg_nodes:
        return seg_nodes[0]
    return None

# Simple function to capture current views
def export_current_views():
    # Capture each of the standard views
    for viewName in ['Red', 'Green', 'Yellow']:
        sliceWidget = layoutManager.sliceWidget(viewName)
        if not sliceWidget:
            print(f"View {viewName} not found.")
            continue
            
        sliceView = sliceWidget.sliceView()
        file_path = os.path.join(output_dir, f"{viewName}_current.png")
        
        try:
            # Grab the view and save as a PNG
            img = sliceView.grab()
            img.save(file_path)
            print(f"Saved: {file_path}")
        except Exception as e:
            print(f"Error saving {file_path}: {str(e)}")
    
    # Also capture the 3D view if needed
    threeDWidget = layoutManager.threeDWidget(0)
    if threeDWidget:
        threeDView = threeDWidget.threeDView()
        file_path = os.path.join(output_dir, "3D_view.png")
        img = threeDView.grab()
        img.save(file_path)
        print(f"Saved: {file_path}")

# Export all slices for the current view orientation
def export_all_slices():
    volumeNode = get_volume_node()
    if not volumeNode:
        print("No volume node found. Please make sure a CT is loaded.")
        return
    
    # Get slice spacing and number of slices from the volume
    spacing = volumeNode.GetSpacing()
    extent = volumeNode.GetImageData().GetExtent()
    numSlices = extent[5] - extent[4] + 1
    sliceSpacing = spacing[2]
    
    print(f"Volume has {numSlices} slices with spacing {sliceSpacing}mm")
    
    # For each slice view (Red, Green, Yellow)
    for viewName in ['Red', 'Green', 'Yellow']:
        sliceWidget = layoutManager.sliceWidget(viewName)
        if not sliceWidget:
            print(f"View {viewName} not found.")
            continue
        
        sliceLogic = sliceWidget.sliceLogic()
        compositeNode = sliceLogic.GetSliceCompositeNode()
        
        # Make sure the volume is selected as background
        compositeNode.SetBackgroundVolumeID(volumeNode.GetID())
        
        # Get slice node
        sliceNode = sliceWidget.mrmlSliceNode()
        sliceController = sliceWidget.sliceController()
        
        # Save current orientation for later restoration
        currentOrientation = sliceNode.GetOrientation()
        
        # Set a known orientation for consistent results
        if viewName == 'Red':
            sliceNode.SetOrientation("Axial")
        elif viewName == 'Yellow':
            sliceNode.SetOrientation("Sagittal") 
        elif viewName == 'Green':
            sliceNode.SetOrientation("Coronal")
        
        # Calculate offset range
        startOffset = sliceLogic.GetSliceOffset() - (numSlices * sliceSpacing) / 2
        endOffset = sliceLogic.GetSliceOffset() + (numSlices * sliceSpacing) / 2
        
        print(f"Exporting {viewName} view slices from {startOffset:.2f}mm to {endOffset:.2f}mm")
        
        # For each slice
        for i in range(numSlices):
            # Calculate offset
            offset = startOffset + i * sliceSpacing
            
            # Set slice offset
            sliceNode.SetSliceOffset(offset)
            sliceController.setSliceOffsetValue(offset)
            
            # Force update
            sliceWidget.repaint()
            
            # Get slice view
            sliceView = sliceWidget.sliceView()
            
            # Create filename
            file_name = f"{viewName}_{sliceNode.GetOrientation()}_{i:03d}.png"
            file_path = os.path.join(output_dir, file_name)
            
            # Save image
            try:
                img = sliceView.grab()
                img.save(file_path)
                print(f"Saved: {file_path}")
            except Exception as e:
                print(f"Error saving {file_path}: {str(e)}")
        
        # Restore original orientation
        sliceNode.SetOrientation(currentOrientation)

# Function to make and export slice views with specific orientations
def export_standard_views():
    volumeNode = get_volume_node()
    if not volumeNode:
        print("No volume node found. Please make sure a CT is loaded.")
        return
    
    # Make sure the segmentation node is visible
    segNode = get_segmentation_node()
    if segNode:
        segNode.SetDisplayVisibility(1)
        print(f"Found segmentation node: {segNode.GetName()}")
    else:
        print("No segmentation node found. Images will not include RTSTRUCT contours.")
    
    # Orientations to export
    orientations = {
        'Red': 'Axial',
        'Yellow': 'Sagittal',
        'Green': 'Coronal'
    }
    
    # For each view and orientation
    for viewName, orientation in orientations.items():
        sliceWidget = layoutManager.sliceWidget(viewName)
        if not sliceWidget:
            continue
        
        # Get slice logic and node
        sliceLogic = sliceWidget.sliceLogic()
        sliceNode = sliceWidget.mrmlSliceNode()
        
        # Set the orientation
        sliceNode.SetOrientation(orientation)
        
        # Make sure the volume is selected as background
        compositeNode = sliceLogic.GetSliceCompositeNode()
        compositeNode.SetBackgroundVolumeID(volumeNode.GetID())
        
        # Force update
        sliceWidget.repaint()
        
        # Save current view
        file_name = f"{viewName}_{orientation}.png"
        file_path = os.path.join(output_dir, file_name)
        
        # Save image
        try:
            img = sliceWidget.sliceView().grab()
            img.save(file_path)
            print(f"Saved: {file_path}")
        except Exception as e:
            print(f"Error saving {file_path}: {str(e)}")

# Function to export a movie of scrolling through slices
def export_slice_scrolling_movie():
    volumeNode = get_volume_node()
    if not volumeNode:
        print("No volume node found. Please make sure a CT is loaded.")
        return
    
    # Get slice spacing and number of slices from the volume
    spacing = volumeNode.GetSpacing()
    extent = volumeNode.GetImageData().GetExtent()
    numSlices = extent[5] - extent[4] + 1
    sliceSpacing = spacing[2]
    
    # For the axial (red) view
    viewName = 'Red'
    sliceWidget = layoutManager.sliceWidget(viewName)
    if not sliceWidget:
        print(f"View {viewName} not found.")
        return
    
    sliceLogic = sliceWidget.sliceLogic()
    compositeNode = sliceLogic.GetSliceCompositeNode()
    
    # Make sure the volume is selected as background
    compositeNode.SetBackgroundVolumeID(volumeNode.GetID())
    
    # Get slice node
    sliceNode = sliceWidget.mrmlSliceNode()
    sliceController = sliceWidget.sliceController()
    
    # Set orientation to axial
    sliceNode.SetOrientation("Axial")
    
    # Calculate offset range
    startOffset = sliceLogic.GetSliceOffset() - (numSlices * sliceSpacing) / 2
    endOffset = sliceLogic.GetSliceOffset() + (numSlices * sliceSpacing) / 2
    
    print(f"Exporting all {numSlices} axial slices from {startOffset:.2f}mm to {endOffset:.2f}mm")
    
    # For each slice
    for i in range(numSlices):
        # Calculate offset
        offset = startOffset + i * sliceSpacing
        
        # Set slice offset
        sliceNode.SetSliceOffset(offset)
        sliceController.setSliceOffsetValue(offset)
        
        # Force update
        sliceWidget.repaint()
        
        # Get slice view
        sliceView = sliceWidget.sliceView()
        
        # Create filename
        file_name = f"Axial_slice_{i:03d}.png"
        file_path = os.path.join(output_dir, file_name)
        
        # Save image
        try:
            img = sliceView.grab()
            img.save(file_path)
            print(f"Saved: {file_path}")
        except Exception as e:
            print(f"Error saving {file_path}: {str(e)}")

# Export specific slice index positions
def export_specific_slices():
    volumeNode = get_volume_node()
    if not volumeNode:
        print("No volume node found. Please make sure a CT is loaded.")
        return
    
    # Get the current patient displayed in the scene
    patientNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSubjectHierarchyNode")
    if patientNode:
        patientName = patientNode.GetAttribute("SubjectHierarchy.PatientID")
        if patientName:
            print(f"Patient: {patientName}")
    
    # Make sure the segmentation node is visible
    segNode = get_segmentation_node()
    if segNode:
        segNode.SetDisplayVisibility(1)
        print(f"Found segmentation node: {segNode.GetName()}")
        
        # Get the segments in the segmentation
        segmentation = segNode.GetSegmentation()
        numSegments = segmentation.GetNumberOfSegments()
        print(f"Segmentation has {numSegments} segments:")
        
        for i in range(numSegments):
            segmentID = segmentation.GetNthSegmentID(i)
            segment = segmentation.GetSegment(segmentID)
            print(f"  - {segment.GetName()}")
    
    # Export key slices in all orientations
    for viewName, orientation in {'Red': 'Axial', 'Yellow': 'Sagittal', 'Green': 'Coronal'}.items():
        sliceWidget = layoutManager.sliceWidget(viewName)
        if not sliceWidget:
            continue
            
        # Set orientation
        sliceNode = sliceWidget.mrmlSliceNode()
        sliceNode.SetOrientation(orientation)
        
        # Get slice logic
        sliceLogic = sliceWidget.sliceLogic()
        
        # Make sure volume is visible
        compositeNode = sliceLogic.GetSliceCompositeNode()
        compositeNode.SetBackgroundVolumeID(volumeNode.GetID())
        
        # Force update
        sliceWidget.repaint()
        
        # For this example, we'll take slices at regular intervals through the volume
        # Adjust step size as needed for your dataset
        steps = 10
        
        # Calculate offset bounds
        sliceOffset = sliceNode.GetSliceOffset()
        boundMin = sliceOffset - 100  # Adjust these bounds as needed
        boundMax = sliceOffset + 100
        stepSize = (boundMax - boundMin) / steps
        
        for i in range(steps + 1):
            offset = boundMin + i * stepSize
            
            # Set slice offset
            sliceNode.SetSliceOffset(offset)
            sliceWidget.sliceController().setSliceOffsetValue(offset)
            
            # Force update
            sliceWidget.repaint()
            
            # Create filename
            file_name = f"{orientation}_step_{i:02d}_offset_{offset:.2f}.png"
            file_path = os.path.join(output_dir, file_name)
            
            # Save image
            try:
                img = sliceWidget.sliceView().grab()
                img.save(file_path)
                print(f"Saved: {file_path}")
            except Exception as e:
                print(f"Error saving {file_path}: {str(e)}")

# Execute the export functions
try:
    print("Starting export of slices with annotations...")
    print(f"Output directory: {output_dir}")
    
    # First try exporting the current views
    print("\nExporting current views...")
    export_current_views()
    
    # Try exporting standard orientation views
    print("\nExporting standard orientation views...")
    export_standard_views()
    
    # Try exporting specific slices
    print("\nExporting specific slices...")
    export_specific_slices()
    
    # Try exporting all slices
    print("\nExporting all slices for each view...")
    export_all_slices()
    
    # Try exporting a scrolling movie of axial slices
    print("\nExporting axial slice movie frames...")
    export_slice_scrolling_movie()
    
    print("\nExport completed. Check the output directory for results.")
    
except Exception as e:
    print(f"An error occurred during export: {str(e)}")
    import traceback
    traceback.print_exc()