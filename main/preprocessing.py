import cv2
import numpy as np
import matplotlib.pyplot as plt


def normalize_image(image, target_mean, target_std):
    """
    Normalize an image to match a target mean and standard deviation.

    Args:
        image (numpy.ndarray): Input image.
        target_mean (float): Target mean intensity.
        target_std (float): Target standard deviation.

    Returns:
        numpy.ndarray: Normalized image.
    """
    current_mean = np.mean(image)
    current_std = np.std(image)
    normalized_image = (image - current_mean) / (current_std + 1e-8)  # Standardize
    normalized_image = normalized_image * target_std + target_mean  # Match target stats
    return np.clip(normalized_image, 0, 255).astype(np.uint8)


def preprocess_images(reference_image, inspected_image):
    """
    Preprocesses the reference and inspected images to normalize lighting and contrast.

    Args:
        reference_image (numpy.ndarray): Reference image (grayscale).
        inspected_image (numpy.ndarray): Inspected image (grayscale).

    Returns:
        tuple: Preprocessed reference and inspected images.
    """
    # Apply CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ref_equalized = clahe.apply(reference_image)
    ins_equalized = clahe.apply(inspected_image)

    # Normalize intensity to match reference statistics
    target_mean = np.mean(ref_equalized)
    target_std = np.std(ref_equalized)
    ref_normalized = normalize_image(ref_equalized, target_mean, target_std)
    ins_normalized = normalize_image(ins_equalized, target_mean, target_std)

    return ref_normalized, ins_normalized


def sift_alignment_and_cropping(reference_image, inspected_image):
    """
    Aligns the inspected image to the reference image using SIFT keypoints and descriptors,
    and crops both images to their overlapping region.

    Args:
        reference_image (numpy.ndarray): Reference image.
        inspected_image (numpy.ndarray): Inspected image to be aligned.

    Returns:
        numpy.ndarray: Cropped and aligned reference image.
        numpy.ndarray: Cropped and aligned inspected image.
    """
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors
    keypoints_ref, descriptors_ref = sift.detectAndCompute(reference_image, None)
    keypoints_ins, descriptors_ins = sift.detectAndCompute(inspected_image, None)

    # Match descriptors using FLANN-based matcher
    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE = 1
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_ref, descriptors_ins, k=2)

    # Apply Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    # Check if enough matches are found
    MIN_MATCH_COUNT = 10
    if len(good_matches) < MIN_MATCH_COUNT:
        raise ValueError("Not enough matches found to compute the homography matrix.")

    # Extract points for homography calculation
    src_pts = np.float32([keypoints_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_ins[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute homography
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    # Warp the inspected image to align with the reference
    aligned_inspected = cv2.warpPerspective(inspected_image, H, (reference_image.shape[1], reference_image.shape[0]),
                                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Compute the intersection of both images
    mask_ref = np.ones_like(reference_image, dtype=np.uint8)
    mask_ins = cv2.warpPerspective(mask_ref, H, (reference_image.shape[1], reference_image.shape[0]))

    intersection_mask = cv2.bitwise_and(mask_ref, mask_ins)
    x, y, w, h = cv2.boundingRect(intersection_mask)

    # Shrink the bounding box slightly to avoid edge artifacts
    padding = 2
    x = max(x + padding, 0)
    y = max(y + padding, 0)
    w = max(w - 2 * padding, 0)
    h = max(h - 2 * padding, 0)

    # Crop both images to the intersection region
    cropped_reference = reference_image[y:y + h, x:x + w]
    cropped_inspected = aligned_inspected[y:y + h, x:x + w]

    # Validate dimensions
    if cropped_reference.shape != cropped_inspected.shape:
        raise ValueError("Aligned images do not have matching dimensions after cropping.")

    return cropped_reference, cropped_inspected


if __name__ == "__main__":
    # Paths to images
    reference_image_path = "data/defective_examples/case1_reference_image.tif"
    inspected_image_path = "data/defective_examples/case1_inspected_image.tif"

    # Load images
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    inspected_image = cv2.imread(inspected_image_path, cv2.IMREAD_GRAYSCALE)

    # Ensure images are loaded
    if reference_image is None:
        raise FileNotFoundError(f"Reference image not found at {reference_image_path}")
    if inspected_image is None:
        raise FileNotFoundError(f"Inspected image not found at {inspected_image_path}")

    # Preprocess images
    reference_image, inspected_image = preprocess_images(reference_image, inspected_image)

    # Perform alignment and cropping
    try:
        cropped_reference, cropped_inspected = sift_alignment_and_cropping(reference_image, inspected_image)
    except ValueError as e:
        print(f"Alignment failed: {e}")
        exit()

    # Visualization of results
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Cropped Aligned Reference Image")
    plt.imshow(cropped_reference, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Cropped Aligned Inspected Image")
    plt.imshow(cropped_inspected, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()