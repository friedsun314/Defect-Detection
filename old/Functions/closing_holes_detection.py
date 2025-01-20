import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_holes(image_path, kernel_size=(5, 5), threshold=10, gaussian_ksize=(5, 5), sigma=1.0):
    """
    Detects "holes" or points of discontinuity in an image using Gaussian smoothing, morphological closing, and a threshold.

    Args:
        image_path (str): Path to the input image.
        kernel_size (tuple): Size of the kernel for morphological closing.
        threshold (int): Threshold value for detecting differences.
        gaussian_ksize (tuple): Size of the Gaussian kernel for smoothing.
        sigma (float): Standard deviation for the Gaussian kernel.

    Returns:
        binary_holes (np.ndarray): Binary image showing detected holes.
    """
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Apply Gaussian smoothing
    smoothed_image = cv2.GaussianBlur(image, gaussian_ksize, sigma)

    # Apply morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    closed_image = cv2.morphologyEx(smoothed_image, cv2.MORPH_CLOSE, kernel)

    # Calculate the absolute difference between the smoothed and closed image
    difference = cv2.absdiff(smoothed_image, closed_image)

    # Threshold the difference to create a binary image
    _, binary_holes = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)

    return binary_holes

# Example usage
if __name__ == "__main__":
    # Input image path
    image_path = "defective_examples/case1_inspected_image.tif"  # Replace with your image path

    # Detect holes
    binary_holes = detect_holes(
        image_path, kernel_size=(5, 5), threshold=10, gaussian_ksize=(5, 5), sigma=1.0
    )

    # Display results
    plt.figure(figsize=(10, 5))
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Detected Holes")
    plt.imshow(binary_holes, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()