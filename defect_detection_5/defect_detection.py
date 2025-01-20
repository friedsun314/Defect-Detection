from preprocessing import normalize_images
from matching import perform_matching, extract_reference_frame
from sliding_window_mse import sliding_window_mse
import cv2
import numpy as np
import matplotlib.pyplot as plt


def initialize_defect_map(image_shape):
    """
    Initialize a defect map with all pixels marked as non-defective (255).

    Args:
        image_shape: Shape of the input image.

    Returns:
        defect_map: Initialized defect map.
    """
    defect_map = np.full(image_shape, 0, dtype=np.uint8)
    plt.figure(figsize=(6, 6))
    plt.title("Initialized Defect Map (All Non-Defective)")
    plt.imshow(defect_map, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    return defect_map


def detect_defects(
    inspected_image, reference_image, frame_sizes, big_frame_size, mse_threshold=1000
):
    """
    Detect defects by comparing inspected and reference images.

    Args:
        inspected_image: Grayscale inspected image.
        reference_image: Grayscale reference image.
        frame_sizes: List of frame sizes for multi-scale analysis.
        big_frame_size: Size of the big frame used for comparison.
        mse_threshold: Minimum MSE to mark a region as defective.

    Returns:
        defect_map: Binary defect map with detected defects.
    """
    defect_map = initialize_defect_map(inspected_image.shape)

    # Perform matching to compute the homography matrix
    H, _, _, _, _ = perform_matching(reference_image, inspected_image)

    for frame_size in frame_sizes:
        print(f"Processing frame size: {frame_size}x{frame_size}")

        for y in range(0, inspected_image.shape[0] - frame_size, frame_size):
            for x in range(0, inspected_image.shape[1] - frame_size, frame_size):
                inspected_frame = inspected_image[y:y+frame_size, x:x+frame_size]

                big_x = max(0, x + frame_size // 2 - big_frame_size // 2)
                big_y = max(0, y + frame_size // 2 - big_frame_size // 2)
                big_x_end = min(inspected_image.shape[1], big_x + big_frame_size)
                big_y_end = min(inspected_image.shape[0], big_y + big_frame_size)
                big_frame = (big_x, big_y, big_x_end - big_x, big_y_end - big_y)

                reference_region = extract_reference_frame(reference_image, H, big_frame)

                if reference_region.shape[0] < frame_size or reference_region.shape[1] < frame_size:
                    continue

                # Use sliding_window_mse to find the lowest MSE and its coordinates
                min_mse, _ = sliding_window_mse(reference_region, inspected_frame)

                print(f"Frame ({x}, {y}) - MSE: {min_mse:.2f}")

                # Mark as defective if MSE exceeds the threshold
                if min_mse < mse_threshold:
                    defect_map[y:y+frame_size, x:x+frame_size] = 255
                    print(f"Updated defect map for frame ({x}, {y}) with frame size {frame_size}")

    return defect_map


# Example usage
if __name__ == "__main__":
    reference_image_path = "data/defective_examples/case1_reference_image.tif"
    inspected_image_path = "data/defective_examples/case1_inspected_image.tif"

    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    inspected_image = cv2.imread(inspected_image_path, cv2.IMREAD_GRAYSCALE)

    if reference_image is None or inspected_image is None:
        raise FileNotFoundError("One or both image paths are incorrect.")

    # Normalize the images
    norm_inspected_image, norm_reference_image = normalize_images(inspected_image, reference_image)

    frame_sizes = [20]
    big_frame_size = 100
    mse_threshold = 80

    defect_map = detect_defects(
        norm_inspected_image, norm_reference_image, frame_sizes, big_frame_size, mse_threshold
    )

    plt.figure(figsize=(10, 5))
    plt.title("Final Defect Map")
    plt.imshow(defect_map, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.show()