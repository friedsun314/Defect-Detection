import cv2
import numpy as np
import matplotlib.pyplot as plt


def perform_matching(reference_image, inspected_image, ratio_threshold=0.7, min_matches=10):
    """
    Perform feature matching between reference and inspected images using SIFT and FLANN.
    Compute the homography matrix if enough matches are found.

    Args:
        reference_image: Grayscale reference image.
        inspected_image: Grayscale inspected image.
        ratio_threshold: Lowe's ratio threshold to filter matches.
        min_matches: Minimum number of matches required to compute the homography matrix.

    Returns:
        H: Homography matrix.
        mask: Inlier mask for the homography matrix.
        keypoints_ref: Keypoints in the reference image.
        keypoints_ins: Keypoints in the inspected image.
        good_matches: List of good matches after Lowe's ratio test.
    """
    sift = cv2.SIFT_create()
    keypoints_ref, descriptors_ref = sift.detectAndCompute(reference_image, None)
    keypoints_ins, descriptors_ins = sift.detectAndCompute(inspected_image, None)

    index_params = dict(algorithm=1, trees=5)  # KDTree
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptors_ref, descriptors_ins, k=2)

    good_matches = [m for m, n in matches if m.distance < ratio_threshold * n.distance]

    if len(good_matches) < min_matches:
        raise ValueError(f"Not enough matches ({len(good_matches)}) to compute homography.")

    src_pts = np.float32([keypoints_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_ins[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    return H, mask, keypoints_ref, keypoints_ins, good_matches


def extract_reference_frame(reference_image, H, big_frame):
    """
    Extract the matched region in the reference image corresponding to the given big frame.

    Args:
        reference_image: Grayscale reference image.
        H: Homography matrix.
        big_frame: Coordinates of the big frame (x, y, width, height) in the inspected image.

    Returns:
        reference_region: Corresponding matched region in the reference image.
    """
    x, y, w, h = big_frame
    points = np.array([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h]
    ], dtype=np.float32).reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(points, H)

    ref_x_min, ref_y_min = np.int32(transformed_points.min(axis=0)[0])
    ref_x_max, ref_y_max = np.int32(transformed_points.max(axis=0)[0])

    reference_region = reference_image[
        max(0, ref_y_min):min(reference_image.shape[0], ref_y_max),
        max(0, ref_x_min):min(reference_image.shape[1], ref_x_max)
    ]

    return reference_region


def visualize_matches(reference_image, inspected_image, keypoints_ref, keypoints_ins, good_matches):
    """
    Visualize the matches between reference and inspected images.

    Args:
        reference_image: Grayscale reference image.
        inspected_image: Grayscale inspected image.
        keypoints_ref: Keypoints in the reference image.
        keypoints_ins: Keypoints in the inspected image.
        good_matches: List of good matches after Lowe's ratio test.
    """
    matched_image = cv2.drawMatches(
        reference_image, keypoints_ref,
        inspected_image, keypoints_ins,
        good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    plt.figure(figsize=(15, 10))
    plt.title("Matches Between Reference and Inspected Images")
    plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


def visualize_reference_frame(reference_image, big_frame, reference_region):
    """
    Visualize the corresponding region in the reference image.

    Args:
        reference_image: Grayscale reference image.
        big_frame: Coordinates of the big frame in the inspected image.
        reference_region: Extracted matched region in the reference image.
    """
    x, y, w, h = big_frame

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Visualize the full reference image with the matched region highlighted
    ax[0].imshow(reference_image, cmap="gray")
    rect = plt.Rectangle((x, y), w, h, edgecolor="green", fill=False, lw=2)
    ax[0].add_patch(rect)
    ax[0].set_title("Reference Image with Matched Region")
    ax[0].axis("off")

    # Show the extracted reference region
    ax[1].imshow(reference_region, cmap="gray")
    ax[1].set_title("Extracted Matched Region in Reference Image")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()


# if __name__ == "__main__":
#     # Paths to images
#     reference_image_path = "data/defective_examples/case1_reference_image.tif"
#     inspected_image_path = "data/defective_examples/case1_inspected_image.tif"

#     # Load images
#     reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
#     inspected_image = cv2.imread(inspected_image_path, cv2.IMREAD_GRAYSCALE)

#     if reference_image is None or inspected_image is None:
#         raise FileNotFoundError("One or both image paths are incorrect.")

#     # Perform matching
#     H, mask, keypoints_ref, keypoints_ins, good_matches = perform_matching(reference_image, inspected_image)

#     # Print homography matrix
#     print("Computed Homography Matrix:")
#     print(H)

#     # Visualize matches
#     visualize_matches(reference_image, inspected_image, keypoints_ref, keypoints_ins, good_matches)

#     # Test extracting a matched frame
#     big_frame = (100, 100, 200, 200)  # Example big frame coordinates in inspected image
#     reference_region = extract_reference_frame(reference_image, H, big_frame)

#     # Visualize the extracted region
#     visualize_reference_frame(reference_image, big_frame, reference_region)

if __name__ == "__main__":
    # Paths to images
    reference_image_path = "data/defective_examples/case1_reference_image.tif"
    inspected_image_path = "data/defective_examples/case1_inspected_image.tif"

    # Load images
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    inspected_image = cv2.imread(inspected_image_path, cv2.IMREAD_GRAYSCALE)

    if reference_image is None or inspected_image is None:
        raise FileNotFoundError("One or both image paths are incorrect.")

    # Perform matching
    H, mask, keypoints_ref, keypoints_ins, good_matches = perform_matching(reference_image, inspected_image)

    # Print homography matrix
    print("Computed Homography Matrix:")
    print(H)

    # Visualize matches
    visualize_matches(reference_image, inspected_image, keypoints_ref, keypoints_ins, good_matches)

    # Warp inspected image using the homography matrix
    warped_inspected_image = cv2.warpPerspective(inspected_image, H, (reference_image.shape[1], reference_image.shape[0]))

    # Compute absolute difference
    difference_image = cv2.absdiff(reference_image, warped_inspected_image)

    # Visualize the difference image
    plt.figure(figsize=(10, 5))
    plt.title("Absolute Difference Between Reference and Warped Inspected Image")
    plt.imshow(difference_image, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # Test extracting a matched frame
    big_frame = (100, 100, 200, 200)  # Example big frame coordinates in inspected image
    reference_region = extract_reference_frame(reference_image, H, big_frame)

    # Visualize the extracted region
    visualize_reference_frame(reference_image, big_frame, reference_region)

