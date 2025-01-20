import numpy as np
import cv2
import matplotlib.pyplot as plt



def adaptive_thresholding(image, threshold_factor=3, kernel_size=5, sigma=1.0):
    """
    Applies Gaussian smoothing followed by adaptive thresholding to extract defect regions.

    Args:
        image (numpy.ndarray): Reconstructed spatial domain image.
        threshold_factor (float): Factor for thresholding (mean - threshold_factor * std).
        kernel_size (int): Size of the Gaussian kernel.
        sigma (float): Standard deviation for the Gaussian kernel.

    Returns:
        numpy.ndarray: Binary mask highlighting defects.
    """
    # Apply Gaussian smoothing
    smoothed_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    # Consider only non-zero pixels for statistics
    non_zero_pixels = smoothed_image[smoothed_image > 0]

    if non_zero_pixels.size == 0:  # Handle empty case
        print("Warning: No non-zero pixels found in the region. Returning an empty mask.")
        return np.zeros_like(image, dtype=np.uint8)

    # Calculate mean and std for non-zero pixels
    mean = np.mean(non_zero_pixels)
    std = np.std(non_zero_pixels)

    # Apply threshold
    threshold = mean + threshold_factor * std
    binary_mask = (smoothed_image < threshold).astype(np.uint8) * 255

    return binary_mask


def postprocess_mask(binary_mask, kernel_size=3):
    """
    Cleans the binary mask using morphological operations to remove noise and fill gaps.

    Args:
        binary_mask (numpy.ndarray): Binary mask.
        kernel_size (int): Kernel size for morphological operations.

    Returns:
        numpy.ndarray: Cleaned binary mask.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Morphological closing: Fill small holes within defect regions
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # Morphological opening: Remove small noise from the edges
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)

    return binary_mask