import numpy as np
import matplotlib.pyplot as plt
from match_features import detect_good_patches  # Import the function for good patches

def initialize_partial_image_map(inspected_image, keypoints_ins, good_matches, patch_size=15):
    """
    Initialize a partial image map where only the pixels corresponding to the
    "good patches" retain their original values, while the rest are set to NaN.

    Args:
        inspected_image: Grayscale inspected image.
        keypoints_ins: Keypoints in the inspected image.
        good_matches: Good matches from the feature matching step.
        patch_size: Half the size of the patch around each keypoint.

    Returns:
        partial_image_map: Image map with original values for good patches and NaN elsewhere.
    """
    # Start with all pixels set to NaN
    partial_image_map = np.full(inspected_image.shape, np.nan, dtype=np.float32)

    # Iterate over good matches
    for match in good_matches:
        kp_ins = keypoints_ins[match.trainIdx]
        x, y = int(kp_ins.pt[0]), int(kp_ins.pt[1])

        # Calculate patch bounds
        x_min = max(0, x - patch_size)
        x_max = min(inspected_image.shape[1], x + patch_size + 1)
        y_min = max(0, y - patch_size)
        y_max = min(inspected_image.shape[0], y + patch_size + 1)

        # Update the partial image map with original values
        partial_image_map[y_min:y_max, x_min:x_max] = inspected_image[y_min:y_max, x_min:x_max]

    # Debug: Check the number of NaN pixels in the partial image map
    print(f"NaN count in partial_image_map after initialization: {np.isnan(partial_image_map).sum()}")

    # Final visualization
    plt.figure(figsize=(6, 6))
    plt.title("Initialized Partial Image Map (Good Patches Retained)")

    # Normalize to match the original image's intensity range
    vmin, vmax = np.nanmin(inspected_image), np.nanmax(inspected_image)
    plt.imshow(partial_image_map, cmap="gray", vmin=vmin, vmax=vmax)

    plt.colorbar(label="Pixel Intensity")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return partial_image_map

if __name__ == "__main__":
    from match_features import detect_good_patches
    import cv2

    # Paths to images
    reference_image_path = "data/defective_examples/case1_reference_image.tif"
    inspected_image_path = "data/defective_examples/case1_inspected_image.tif"

    # Load images
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    inspected_image = cv2.imread(inspected_image_path, cv2.IMREAD_GRAYSCALE)

    # Detect good patches
    good_patches, keypoints_ref, keypoints_ins, good_matches = detect_good_patches(
        reference_image, inspected_image, patch_size=15
    )

    # Initialize partial image map
    partial_image_map = initialize_partial_image_map(inspected_image, keypoints_ins, good_matches, patch_size=15)