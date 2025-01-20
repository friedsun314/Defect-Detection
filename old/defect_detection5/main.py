import cv2
import numpy as np
from match_features import detect_good_patches
from initialize_partial_image_map import initialize_partial_image_map
from iterative_partial_image_update import iterative_partial_image_update
import matplotlib.pyplot as plt

# Paths to the reference and inspected images
reference_image_path = "data/defective_examples/case1_reference_image.tif"
inspected_image_path = "data/defective_examples/case1_inspected_image.tif"

def main():
    # Load images in grayscale
    print("Loading images...")
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    inspected_image = cv2.imread(inspected_image_path, cv2.IMREAD_GRAYSCALE)

    if reference_image is None or inspected_image is None:
        raise FileNotFoundError("Could not load one or both images. Check the file paths.")

    # Step 1: Detect good patches using feature matching
    print("Detecting good patches...")
    good_patches, keypoints_ref, keypoints_ins, good_matches = detect_good_patches(
        reference_image, inspected_image, ratio_threshold=0.7, patch_size=15
    )

    # Step 2: Initialize the partial image map
    print("Initializing partial image map...")
    partial_image_map = initialize_partial_image_map(
        inspected_image, keypoints_ins, good_matches, patch_size=15
    )

    # Step 3: Iteratively update the partial image map using matchTemplate
    print("Iteratively updating the partial image map...")
    updated_partial_image = iterative_partial_image_update(
        inspected_image, partial_image_map, min_patch_size=5, max_patch_size=100, similarity_threshold=0.6, mse_threshold=93, max_repeats=2
    )

    # Step 4: Visualize results
    print("Visualizing results...")
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("Reference Image")
    plt.imshow(reference_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Inspected Image")
    plt.imshow(inspected_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Updated Partial Image")
    plt.imshow(updated_partial_image, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()