import os
import cv2
import numpy as np

def contains_exact_green(image_rgb, target_color=(0, 255, 0)):
    return np.any(np.all(image_rgb == target_color, axis=-1))

def contains_tolerant_green(image_rgb, target_color=(0, 255, 0), tolerance=10):
    diff = np.abs(image_rgb - np.array(target_color))
    mask = np.all(diff <= tolerance, axis=-1)
    return np.any(mask)

def get_unique_colors(image_rgb):
    # Reshape to a list of RGB tuples
    pixels = image_rgb.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)
    return [tuple(color) for color in unique_colors]

def clean_dataset(image_folder, mask_folder):
    image_files = set(os.listdir(image_folder))
    mask_files = os.listdir(mask_folder)

    deleted_count = 0

    for mask_file in mask_files:
        if not mask_file.lower().endswith(".png"):
            continue

        mask_path = os.path.join(mask_folder, mask_file)
        image_path = os.path.join(image_folder, mask_file)

        # Load mask image
        mask_bgr = cv2.imread(mask_path)
        if mask_bgr is None:
            print(f"Could not read: {mask_path}")
            continue

        mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)

        # Check for exact green
        if contains_exact_green(mask_rgb):
            continue  # Green found, keep file

        # Green not found, print colors before tolerance
        print(f"\nNo exact green in: {mask_file}")
        unique_colors = get_unique_colors(mask_rgb)
        print(f"Unique RGB colors in mask: {unique_colors[:15]}{' ...' if len(unique_colors) > 15 else ''}")

        # Apply tolerant green match
        if contains_tolerant_green(mask_rgb):
            print(f"✓ Green found with tolerance in: {mask_file}")
            continue  # Green found with tolerance

        # No green found even with tolerance → delete
        print(f"✗ Deleting mask and corresponding image: {mask_file}")
        os.remove(mask_path)
        deleted_count += 1

        if os.path.exists(image_path):
            os.remove(image_path)

    print(f"\nCleanup complete. Deleted {deleted_count} mask(s) and image(s).")

# Set your Windows-style paths (raw string or double backslashes)
image_folder = r"images"
mask_folder = r"masks"

clean_dataset(image_folder, mask_folder)
