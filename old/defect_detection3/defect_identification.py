import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys


def grow_good_pixels(inspected_image, good_key_points, good_descriptors, patch_size=15, similarity_threshold=50):
    """
    Iteratively grow the set of good pixels by finding new key points similar to the good key points.

    Args:
        inspected_image: The inspected grayscale image.
        good_key_points: List of current good key points.
        good_descriptors: Descriptors corresponding to the good key points.
        patch_size: Size of the patches to extract around each key point.
        similarity_threshold: Threshold for similarity to classify a key point as non-defective.

    Returns:
        good_pixels_mask: Binary mask marking good (non-defective) pixels.
        updated_good_key_points: Updated list of good key points.
        updated_good_descriptors: Updated list of descriptors for good key points.
    """
    good_pixels_mask = np.zeros_like(inspected_image, dtype=np.uint8)
    
    # Mark the good pixels based on initial good key points
    for kp in good_key_points:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        good_pixels_mask[
            max(0, y-patch_size):min(inspected_image.shape[0], y+patch_size),
            max(0, x-patch_size):min(inspected_image.shape[1], x+patch_size)
        ] = 255  # Mark as good pixels

    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    iteration = 0

    while True:
        iteration += 1
        print(f"Iteration {iteration}: Growing good pixels...")

        # Find new key points in the "defective image" (remaining regions)
        remaining_image = cv2.bitwise_and(inspected_image, cv2.bitwise_not(good_pixels_mask))
        new_key_points, new_descriptors = sift.detectAndCompute(remaining_image, None)

        if new_descriptors is None or len(new_key_points) == 0:
            print("No new key points found. Stopping iteration.")
            break

        # Check if there are enough good descriptors
        if good_descriptors is None or len(good_descriptors) < 2:
            print("Not enough good descriptors for matching. Stopping iteration.")
            break

        # Match new key points to good key points using FLANN
        index_params = dict(algorithm=1, trees=5)  # Using KDTree
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(new_descriptors, good_descriptors, k=2)

        # Apply Lowe's ratio test and similarity threshold
        new_good_key_points = []
        new_good_descriptors = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:  # Lowe's ratio test
                new_good_key_points.append(new_key_points[m.queryIdx])
                new_good_descriptors.append(new_descriptors[m.queryIdx])

        if len(new_good_key_points) == 0:
            print("No additional good pixels added. Stopping iteration.")
            break

        # Update the good pixels mask with the newly added key points
        for kp in new_good_key_points:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            good_pixels_mask[
                max(0, y-patch_size):min(inspected_image.shape[0], y+patch_size),
                max(0, x-patch_size):min(inspected_image.shape[1], x+patch_size)
            ] = 255  # Mark as good pixels

        # Update the set of good key points and descriptors
        good_key_points.extend(new_good_key_points)
        good_descriptors = np.vstack((good_descriptors, np.array(new_good_descriptors)))

    return good_pixels_mask

def identify_defects_iterative(inspected_image, good_key_points, good_descriptors, patch_size=15, similarity_threshold=50):
    """
    Identify defects in the inspected image iteratively by growing the set of good pixels.

    Args:
        inspected_image: The inspected grayscale image.
        good_key_points: Initial set of good key points.
        good_descriptors: Descriptors corresponding to the good key points.
        patch_size: Size of the patches to extract around each key point.
        similarity_threshold: Threshold for similarity to classify a key point as non-defective.

    Returns:
        defect_map: Binary map marking defective pixels.
    """
    # Call grow_good_pixels to iteratively expand the good pixels set
    good_pixels_mask = grow_good_pixels(
        inspected_image, good_key_points, good_descriptors, patch_size, similarity_threshold
    )
    
    # The defect map is the complement of the good pixels mask
    defect_map = cv2.bitwise_not(good_pixels_mask)
    return defect_map

if __name__ == "__main__":
    # Load the inspected image
    inspected_image = np.zeros((100, 100), dtype=np.uint8)
    inspected_image[40:60, 40:60] = 255  # Add a white defect

    # Initialize synthetic good key points
    good_key_points = [cv2.KeyPoint(30, 30, 10)]
    good_descriptors = np.random.rand(1, 128).astype(np.float32)  # Mock descriptors

    # Identify defects iteratively
    defect_map = identify_defects_iterative(
        inspected_image, good_key_points, good_descriptors, patch_size=10, similarity_threshold=50
    )

    # Visualize the defect map
    plt.figure(figsize=(8, 8))
    plt.title("Defected Pixels Map")
    plt.imshow(defect_map, cmap="gray")
    plt.axis("off")
    plt.show()