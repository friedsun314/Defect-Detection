import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import shift
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
from collections import Counter
import matplotlib.pyplot as plt


def preprocess_image(image, sigma=2):
    """
    Preprocess an image by applying Gaussian smoothing and normalization.

    Parameters:
        image (np.ndarray): The input image.
        sigma (float): The standard deviation for Gaussian kernel. Higher values mean more smoothing.

    Returns:
        np.ndarray: The preprocessed image.
    """
    # Apply Gaussian smoothing
    smoothed_image = gaussian_filter(image, sigma=sigma)

    # Normalize the image to the range [0, 1]
    normalized_image = rescale_intensity(smoothed_image, in_range='image', out_range=(0, 1))

    return normalized_image


def align_images_majority_vote_debug(ref_image, inspected_image, n_frames=50, top_percent=5):
    """
    Align the inspected image to the reference image using a robust, frame-based approach.
    Includes visualizations of random frames and their matches.

    Parameters:
        ref_image (np.ndarray): The reference image (2D array).
        inspected_image (np.ndarray): The inspected image (2D array).
        n_frames (int): Number of random frames to sample from the inspected image.
        top_percent (float): Percentage of top pairs to consider for voting.

    Returns:
        best_shift (tuple): The shift (y_shift, x_shift) determined by majority vote.
    """
    ref_h, ref_w = ref_image.shape
    insp_h, insp_w = inspected_image.shape
    all_pairs = []

    print(f"Reference Image Shape: {ref_image.shape}")
    print(f"Inspected Image Shape: {inspected_image.shape}\n")

    # Randomly sample n frames from the inspected image
    for frame_idx in range(n_frames):
        # Randomly determine frame size with a minimum constraint
        min_frame_size = 20
        max_frame_size = 100
        frame_h = np.random.randint(min_frame_size, max_frame_size)
        frame_w = np.random.randint(min_frame_size, max_frame_size)

        # Randomly sample a frame from the inspected image
        start_y = np.random.randint(0, insp_h - frame_h + 1)
        start_x = np.random.randint(0, insp_w - frame_w + 1)
        inspected_frame = inspected_image[start_y:start_y + frame_h, start_x:start_x + frame_w]

        print(f"Frame {frame_idx + 1}:")
        print(f"  Frame Size: ({frame_h}, {frame_w})")
        print(f"  Inspected Frame (Coordinates: ({start_y}, {start_x}))")

        # Track the best match
        best_match_coords = None
        best_match_frame = None
        min_distance = float('inf')

        # Iterate over all possible frames of the same size in the reference image
        for ref_y in range(ref_h - frame_h + 1):
            for ref_x in range(ref_w - frame_w + 1):
                ref_frame = ref_image[ref_y:ref_y + frame_h, ref_x:ref_x + frame_w]
                distance = np.linalg.norm(inspected_frame - ref_frame, ord='fro')

                # Track the best match
                if distance < min_distance:
                    min_distance = distance
                    best_match_coords = (ref_y, ref_x)
                    best_match_frame = ref_frame

                all_pairs.append(((start_y, start_x), (ref_y, ref_x), distance))

        # Visualize the inspected frame and its best match
        # if best_match_frame is not None:
        #     ref_y, ref_x = best_match_coords
        #     print(f"  Best Match (Ref Coord: ({ref_y}, {ref_x}), Distance: {min_distance:.4f})")

        #     plt.figure(figsize=(12, 4))
        #     plt.suptitle(f"Frame {frame_idx + 1}: Inspected vs. Best Match", fontsize=14)

        #     # Inspected Frame
        #     plt.subplot(1, 3, 1)
        #     plt.title("Inspected Frame")
        #     plt.imshow(inspected_frame, cmap="gray")
        #     plt.axis("off")

        #     # Best-Matching Frame
        #     plt.subplot(1, 3, 2)
        #     plt.title("Best-Matching Frame (Reference)")
        #     plt.imshow(best_match_frame, cmap="gray")
        #     plt.axis("off")

        #     # Overlay
        #     plt.subplot(1, 3, 3)
        #     plt.title("Overlay")
        #     plt.imshow(inspected_frame, cmap="gray", alpha=0.5)
        #     plt.imshow(best_match_frame, cmap="jet", alpha=0.5)
        #     plt.axis("off")

        #     plt.show()

    # Sort all pairs by distance and keep the top 5%
    all_pairs.sort(key=lambda x: x[2])  # Sort by distance
    top_k = max(1, len(all_pairs) * top_percent // 100)
    top_pairs = all_pairs[:top_k]

    print("\nTop 5% Pairs:")
    for (start_y, start_x), (ref_y, ref_x), distance in top_pairs:
        print(f"  Inspected Coord: ({start_y}, {start_x}), Ref Coord: ({ref_y}, {ref_x}), Distance: {distance:.4f}")

    # Calculate suggested shifts from the top pairs
    suggested_shifts = [(ref_y - start_y, ref_x - start_x) for (start_y, start_x), (ref_y, ref_x), _ in top_pairs]

    # Use majority voting to determine the best shift
    print("\nSuggested Shifts:")
    print(suggested_shifts)
    best_shift = Counter(suggested_shifts).most_common(1)[0][0]
    print(f"Majority Vote: Best Shift = {best_shift}")
    return best_shift


def test_with_actual_images():
    # Load the images
    inspected_image_path = "defect_detection/data/defective_examples/case1_inspected_image.tif"
    reference_image_path = "defect_detection/data/defective_examples/case1_reference_image.tif"

    inspected_image = imread(inspected_image_path)
    reference_image = imread(reference_image_path)

    # Ensure images are grayscale
    if inspected_image.ndim == 3 and inspected_image.shape[-1] == 3:
        inspected_image = rgb2gray(inspected_image)
    if reference_image.ndim == 3 and reference_image.shape[-1] == 3:
        reference_image = rgb2gray(reference_image)

    # Preprocess the images
    sigma = 2  # Standard deviation for Gaussian smoothing
    print("\nPreprocessing images...")
    inspected_image = preprocess_image(inspected_image, sigma=sigma)
    reference_image = preprocess_image(reference_image, sigma=sigma)

    print("Preprocessed Inspected Image:")
    print(inspected_image)
    print("Preprocessed Reference Image:")
    print(reference_image)

    # Align using the improved logic with debug
    best_shift = align_images_majority_vote_debug(reference_image, inspected_image)
    print("\nFinal Best Shift (Majority Vote):", best_shift)

    # Apply the shift to align the inspected image
    aligned_image = np.roll(inspected_image, shift=(-best_shift[0], -best_shift[1]), axis=(0, 1))

    # Visualizations
    plt.figure(figsize=(10, 8))

    # Original Inspected Image
    plt.subplot(2, 3, 1)
    plt.title("Inspected Image (Original)")
    plt.imshow(imread(inspected_image_path), cmap="gray")
    plt.axis("off")

    # Original Reference Image
    plt.subplot(2, 3, 2)
    plt.title("Reference Image (Original)")
    plt.imshow(imread(reference_image_path), cmap="gray")
    plt.axis("off")

    # Preprocessed Inspected Image
    plt.subplot(2, 3, 3)
    plt.title("Inspected Image (Preprocessed)")
    plt.imshow(inspected_image, cmap="gray")
    plt.axis("off")

    # Preprocessed Reference Image
    plt.subplot(2, 3, 4)
    plt.title("Reference Image (Preprocessed)")
    plt.imshow(reference_image, cmap="gray")
    plt.axis("off")

    # Aligned Inspected Image
    plt.subplot(2, 3, 5)
    plt.title("Aligned Inspected Image")
    plt.imshow(aligned_image, cmap="gray")
    plt.axis("off")

    # Overlay
    plt.subplot(2, 3, 6)
    plt.title("Overlay: Reference vs. Aligned")
    plt.imshow(reference_image, cmap="gray", alpha=0.5)
    plt.imshow(aligned_image, cmap="jet", alpha=0.5)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_with_actual_images()