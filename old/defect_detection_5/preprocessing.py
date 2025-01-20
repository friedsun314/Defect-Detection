import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)

def normalize_images(inspected_image, reference_image, range_min=0, range_max=255):
    norm_inspected = cv2.normalize(inspected_image, None, range_min, range_max, cv2.NORM_MINMAX)
    norm_reference = cv2.normalize(reference_image, None, range_min, range_max, cv2.NORM_MINMAX)
    return norm_inspected, norm_reference

def preprocess_image(image, gaussian_kernel_size=5, morph_kernel_size=5):
    smoothed = cv2.GaussianBlur(image, (gaussian_kernel_size, gaussian_kernel_size), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    opened = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed

if __name__ == "__main__":
    # Paths to images
    reference_image_path = "data/defective_examples/case1_reference_image.tif"
    inspected_image_path = "data/defective_examples/case1_inspected_image.tif"

    # Load images
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    inspected_image = cv2.imread(inspected_image_path, cv2.IMREAD_GRAYSCALE)

    if reference_image is None:
        raise FileNotFoundError(f"Reference image not found at {reference_image_path}")
    if inspected_image is None:
        raise FileNotFoundError(f"Inspected image not found at {inspected_image_path}")

    # Perform normalization
    logging.info("Normalizing images...")
    norm_inspected_image, norm_reference_image = normalize_images(inspected_image, reference_image)

    # Apply preprocessing
    logging.info("Preprocessing images...")
    preprocessed_inspected_image = preprocess_image(norm_inspected_image)
    preprocessed_reference_image = preprocess_image(norm_reference_image)

    # Visualizations
    logging.info("Displaying preprocessed images...")
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Preprocessed Reference Image")
    plt.imshow(preprocessed_reference_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Preprocessed Inspected Image")
    plt.imshow(preprocessed_inspected_image, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Save preprocessed images
    cv2.imwrite(reference_image_path.replace(".tif", "_preprocessed.tif"), preprocessed_reference_image)
    cv2.imwrite(inspected_image_path.replace(".tif", "_preprocessed.tif"), preprocessed_inspected_image)