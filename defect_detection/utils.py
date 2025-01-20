import numpy as np
from PIL import Image

def load_image(image_path):
    """
    Load an image from the specified path, convert it to grayscale, and normalize to [0, 1].

    Parameters:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: Grayscale image normalized to [0, 1].
    """
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    return np.array(image, dtype=np.float32) / 255.0