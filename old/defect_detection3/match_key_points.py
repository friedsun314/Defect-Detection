import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

# Adjust the path to ensure imports work correctly
sys.path.append("..")  # Adjust this path as needed for your project structure

# Import the detect_key_points function from detect_key_points.py
from detect_key_points import detect_key_points

def match_key_points_flann(inspected_image, reference_image, keypoints_inspected, keypoints_reference, descriptors_inspected, descriptors_reference, top_percent=5):
    """
    Match key points using FLANN-based matching.

    Args:
        inspected_image: The inspected grayscale image.
        reference_image: The reference grayscale image.
        keypoints_inspected: Key points from the inspected image.
        keypoints_reference: Key points from the reference image.
        descriptors_inspected: Descriptors from the inspected image.
        descriptors_reference: Descriptors from the reference image.
        top_percent: Percentage of matches to retain based on similarity score.

    Returns:
        filtered_matches: List of matched key points and their corresponding points in the reference image.
    """
    if descriptors_inspected is None or descriptors_reference is None:
        raise ValueError("Descriptors cannot be None. Ensure key points and descriptors are correctly extracted.")

    # Define FLANN parameters
    index_params = dict(algorithm=1, trees=5)  # Using KDTree
    search_params = dict(checks=50)  # Number of checks
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match descriptors
    matches = flann.knnMatch(descriptors_inspected, descriptors_reference, k=2)

    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Extract matched key points
    matched_points = []
    scores = []
    for match in good_matches:
        kp_ins = keypoints_inspected[match.queryIdx]
        kp_ref = keypoints_reference[match.trainIdx]
        matched_points.append((kp_ins, kp_ref))
        scores.append(match.distance)

    # Select top matches based on distance
    threshold_score = np.percentile(scores, 100 - top_percent)
    filtered_matches = [(kp_ins, kp_ref) for (kp_ins, kp_ref), score in zip(matched_points, scores) if score <= threshold_score]
    
    return filtered_matches


def visualize_matches_one_by_one(inspected_image, reference_image, matched_points):
    """
    Visualize matched key points between inspected and reference images one by one.
    
    Args:
        inspected_image: The inspected grayscale image.
        reference_image: The reference grayscale image.
        matched_points: List of matched key points and their corresponding points in the reference image.
    """
    inspected_with_matches = cv2.cvtColor(inspected_image, cv2.COLOR_GRAY2BGR)
    reference_with_matches = cv2.cvtColor(reference_image, cv2.COLOR_GRAY2BGR)
    
    for i, (kp_ins, kp_ref) in enumerate(matched_points):
        # Get key point coordinates
        inspected_point = (int(kp_ins.pt[0]), int(kp_ins.pt[1]))
        reference_point = (int(kp_ref.pt[0]), int(kp_ref.pt[1]))

        # Draw the key point in the inspected image
        temp_inspected = inspected_with_matches.copy()
        cv2.circle(temp_inspected, inspected_point, 10, (0, 255, 0), 2)
        cv2.putText(temp_inspected, f"Match {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw the matched key point in the reference image
        temp_reference = reference_with_matches.copy()
        cv2.circle(temp_reference, reference_point, 10, (0, 255, 0), 2)
        cv2.putText(temp_reference, f"Match {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Plot the images side by side
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title(f"Inspected Image - Match {i+1}")
        plt.imshow(cv2.cvtColor(temp_inspected, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title(f"Reference Image - Match {i+1}")
        plt.imshow(cv2.cvtColor(temp_reference, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.tight_layout()
        plt.show()

        # Pause after each match for validation
        input(f"Press Enter to see the next match (Match {i+1}/{len(matched_points)}).")

if __name__ == "__main__":
    # Paths to images
    inspected_image_path = "data/defective_examples/case1_inspected_image.tif"
    reference_image_path = "data/defective_examples/case1_reference_image.tif"

    # Load images
    inspected_image = cv2.imread(inspected_image_path, cv2.IMREAD_GRAYSCALE)
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)

    if inspected_image is None:
        raise FileNotFoundError(f"Failed to load inspected image from {inspected_image_path}")
    if reference_image is None:
        raise FileNotFoundError(f"Failed to load reference image from {reference_image_path}")

    # Detect key points and descriptors in both images using the imported function
    method = "SIFT"  # You can also try "ORB"
    keypoints_ins, descriptors_ins = detect_key_points(inspected_image, method=method)
    keypoints_ref, descriptors_ref = detect_key_points(reference_image, method=method)

    # Match key points using FLANN-based matcher
    top_percent = 5  # Retain top 5% of matches
    matched_points = match_key_points_flann(inspected_image, reference_image, keypoints_ins, keypoints_ref, descriptors_ins, descriptors_ref, top_percent=top_percent)

    # Visualize the matched points
    visualize_matches_one_by_one(inspected_image, reference_image, matched_points)