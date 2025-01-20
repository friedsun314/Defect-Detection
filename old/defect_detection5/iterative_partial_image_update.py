import cv2
import matplotlib.pyplot as plt
import numpy as np


def calculate_mse(patch1, patch2):
    """
    Calculate the Mean Squared Error (MSE) between two patches.

    Args:
        patch1: First patch as a 2D numpy array.
        patch2: Second patch as a 2D numpy array.

    Returns:
        mse: Mean Squared Error between the patches.
    """
    return np.mean((patch1 - patch2) ** 2)




def get_patches_with_missing_pixels(partial_image, patch_size):
    """
    Extract all patches of a given size from the partial_image that contain missing pixels (NaN).

    Args:
        partial_image: 2D numpy array with NaN values for missing regions.
        patch_size: Size of the square patch.

    Returns:
        List of tuples: [(y, x, patch), ...] where y, x are the top-left coordinates of the patch
        and patch is the extracted patch.
    """
    height, width = partial_image.shape
    patches = []

    for y in range(0, height - patch_size + 1, 5):
        for x in range(0, width - patch_size + 1, 5):
            patch = partial_image[y:y + patch_size, x:x + patch_size]
            if np.isnan(patch).any():
                patches.append((y, x, patch))

    return patches

def iterative_partial_image_update(
    inspected_image, partial_image, min_patch_size=15, max_patch_size=30, similarity_threshold=0.8, mse_threshold=100, max_repeats=3
):
    """
    Iteratively update the partial_image using the inspected_image with systematic scanning.
    Starts with the largest patch size and gradually reduces to the smallest patch size.

    Args:
        inspected_image: Grayscale image to search for matches.
        partial_image: 2D numpy array with NaN values for missing regions.
        min_patch_size: Minimum patch size for scanning.
        max_patch_size: Maximum patch size for scanning.
        similarity_threshold: Threshold for template matching (normalized cross-correlation).
        mse_threshold: Maximum allowable Mean Squared Error for a patch to be inserted.
        max_repeats: Number of times to repeat the process with varying patch sizes.

    Returns:
        Updated partial_image.
    """
    for repeat in range(max_repeats):
        print(f"Repeat {repeat + 1}/{max_repeats}:")

        # Start with the largest patch size and decrease to the smallest
        for patch_size in range(max_patch_size, min_patch_size - 1, -5):
            print(f"  Processing with patch size: {patch_size}")

            # Extract patches with missing pixels
            patches = get_patches_with_missing_pixels(partial_image, patch_size)
            print(f"  Found {len(patches)} patches with missing pixels.")

            for y, x, partial_patch in patches:
                inspected_patch = inspected_image[y:y + patch_size, x:x + patch_size]

                # Replace NaN with a high value in the partial image
                masked_partial_image = np.copy(partial_image)
                masked_partial_image[np.isnan(masked_partial_image)] = 1e6

                # Perform template matching
                match_result = cv2.matchTemplate(
                    masked_partial_image.astype(np.float32),
                    inspected_patch.astype(np.float32),
                    method=cv2.TM_CCOEFF_NORMED
                )

                _, max_val, _, max_loc = cv2.minMaxLoc(match_result)

                if max_val >= similarity_threshold:
                    matched_x, matched_y = max_loc

                    # Extract the matched patch from the partial image
                    matched_patch = inspected_image[
                        matched_y:matched_y + patch_size, matched_x:matched_x + patch_size
                    ]

                    # Calculate MSE between the inspected patch and the matched patch
                    mse = calculate_mse(inspected_patch, matched_patch)
                    print(f"    Max value: {max_val}, MSE: {mse}")

                    # If MSE is below the threshold, update the partial image
                    if mse <= mse_threshold:
                        partial_image[matched_y:matched_y + patch_size, matched_x:matched_x + patch_size] = inspected_image[
                            matched_y:matched_y + patch_size, matched_x:matched_x + patch_size
                        ]
                        print(f"    Patch inserted at ({matched_x}, {matched_y}).")

    return partial_image


# Test and Visualization
if __name__ == "__main__":
    # Create synthetic data for testing
    inspected_image = np.random.rand(100, 100) * 255
    partial_image = np.copy(inspected_image)
    partial_image[30:50, 30:50] = np.nan
    partial_image[70:90, 70:90] = np.nan

    # Visualization of initial state
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Inspected Image")
    plt.imshow(inspected_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Partial Image (Initial)")
    plt.imshow(partial_image, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # Run the iterative update
    updated_partial_image = iterative_partial_image_update(
        inspected_image, partial_image, min_patch_size=10, max_patch_size=100, similarity_threshold=0.9, max_repeats=3
    )

    # Visualization of updated map
    plt.figure(figsize=(6, 6))
    plt.title("Updated Partial Image")
    plt.imshow(updated_partial_image, cmap="gray")
    plt.axis("off")
    plt.show()