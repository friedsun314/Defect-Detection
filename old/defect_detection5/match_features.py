import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_good_patches(reference_image, inspected_image, ratio_threshold=0.7, patch_size=15):
    """
    Detect good patches between the reference and inspected images using SIFT and FLANN.
    Expand keypoints into patches.

    Args:
        reference_image: Grayscale reference image.
        inspected_image: Grayscale inspected image.
        ratio_threshold: Lowe's ratio threshold for good matches.
        patch_size: Half the size of the patch around each keypoint (total size is 2 * patch_size + 1).

    Returns:
        good_patches: List of tuples containing (patch_ref, patch_ins) for the reference and inspected patches.
        keypoints_ref: Keypoints in the reference image.
        keypoints_ins: Keypoints in the inspected image.
        good_matches: Good matches after Lowe's ratio test.
    """
    sift = cv2.SIFT_create()
    keypoints_ref, descriptors_ref = sift.detectAndCompute(reference_image, None)
    keypoints_ins, descriptors_ins = sift.detectAndCompute(inspected_image, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptors_ref, descriptors_ins, k=2)
    good_matches = [m for m, n in matches if m.distance < ratio_threshold * n.distance]

    good_patches = []
    for match in good_matches:
        # Get matched keypoints
        kp_ref = keypoints_ref[match.queryIdx]
        kp_ins = keypoints_ins[match.trainIdx]

        x_ref, y_ref = int(kp_ref.pt[0]), int(kp_ref.pt[1])
        x_ins, y_ins = int(kp_ins.pt[0]), int(kp_ins.pt[1])

        # Extract patches around the keypoints
        patch_ref = reference_image[
            max(0, y_ref-patch_size):min(reference_image.shape[0], y_ref+patch_size+1),
            max(0, x_ref-patch_size):min(reference_image.shape[1], x_ref+patch_size+1)
        ]
        patch_ins = inspected_image[
            max(0, y_ins-patch_size):min(inspected_image.shape[0], y_ins+patch_size+1),
            max(0, x_ins-patch_size):min(inspected_image.shape[1], x_ins+patch_size+1)
        ]

        # Only include valid patches (those with the correct size)
        if patch_ref.shape == (2 * patch_size + 1, 2 * patch_size + 1) and \
           patch_ins.shape == (2 * patch_size + 1, 2 * patch_size + 1):
            good_patches.append((patch_ref, patch_ins))

    return good_patches, keypoints_ref, keypoints_ins, good_matches

def visualize_good_patches(good_patches):
    """
    Visualize a few good patches for validation.

    Args:
        good_patches: List of tuples containing (patch_ref, patch_ins).
    """
    num_patches = min(len(good_patches), 10)  # Limit to 10 patches for visualization
    fig, axes = plt.subplots(num_patches, 2, figsize=(10, num_patches * 2))

    for i, (patch_ref, patch_ins) in enumerate(good_patches[:num_patches]):
        axes[i, 0].imshow(patch_ref, cmap="gray")
        axes[i, 0].set_title(f"Reference Patch {i+1}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(patch_ins, cmap="gray")
        axes[i, 1].set_title(f"Inspected Patch {i+1}")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    reference_image_path = "data/defective_examples/case1_reference_image.tif"
    inspected_image_path = "data/defective_examples/case1_inspected_image.tif"

    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    inspected_image = cv2.imread(inspected_image_path, cv2.IMREAD_GRAYSCALE)

    good_patches, keypoints_ref, keypoints_ins, good_matches = detect_good_patches(
        reference_image, inspected_image, patch_size=15
    )
    visualize_good_patches(good_patches)