import os
import cv2
import numpy as np

def load_images_from_folder(folder_path):
    """
    Reads all images in a folder sequentially and converts them to NumPy arrays using OpenCV.

    Args:
        folder_path (str): Path to the folder containing images.

    Returns:
        list: A list of NumPy arrays, one for each image.
    """
    # List all files in the folder
    for filename in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        try:
            # Read the image using OpenCV
            img = cv2.imread(file_path)
            if img is not None:
                yield np.asarray(img)
            else:
                print(f"Skipping {file_path}: Not a valid image.")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

# Example usage
folder_path = "/path/to/your/image/folder"
image_arrays = load_images_from_folder(folder_path)