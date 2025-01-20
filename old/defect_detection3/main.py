import cv2
import numpy as np
import matplotlib.pyplot as plt
from defect_identification import grow_good_pixels, identify_defects_iterative
from detect_key_points import detect_key_points
import sys

if __name__ == "__main__":
    # Paths to actual images
    inspected_image_path = "data/defective_examples/case1_inspected_image.tif"
    reference_image_path = "data/defective_examples/case1_reference_image.tif"

    # Load images
    inspected_image = cv2.imread(inspected_image_path, cv2.IMREAD_GRAYSCALE)
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)

    if inspected_image is None:
        raise FileNotFoundError(f"Failed to load inspected image from {inspected_image_path}")
    if reference_image is None:
        raise FileNotFoundError(f"Failed to load reference image from {reference_image_path}")

    # Step 1: Detect initial good key points from the reference image
    method = "SIFT"  # You can also try "ORB" if needed
    keypoints_ins, descriptors_ins = detect_key_points(inspected_image, method=method)
    keypoints_ref, descriptors_ref = detect_key_points(reference_image, method=method)

    # Step 2: Match key points between inspected and reference images
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptors_ins, descriptors_ref, k=2)

    # Apply Lowe's ratio test to filter matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Extract good key points and descriptors from the inspected image
    good_key_points = [keypoints_ins[m.queryIdx] for m in good_matches]
    good_descriptors = np.array([descriptors_ins[m.queryIdx] for m in good_matches])

    # Step 3: Iteratively identify defects in the inspected image
    patch_size = 15  # Size of patches around each key point
    similarity_threshold = 50  # Threshold for similarity metric
    defect_map = identify_defects_iterative(
        inspected_image, good_key_points, good_descriptors, patch_size=patch_size, similarity_threshold=similarity_threshold
    )

    # Step 4: Visualize the defect map
    plt.figure(figsize=(12, 12))
    plt.title("Defected Pixels Map")
    plt.imshow(defect_map, cmap="gray")
    plt.axis("off")
    plt.show()