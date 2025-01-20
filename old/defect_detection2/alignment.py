import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/guy/Desktop/Muze AI")
from defect_detection2.shift import pad_and_shift_image


def center_reference(reference, inspected_shape):
    """
    Center the reference image in a larger matrix of zeros.

    Parameters:
        reference (np.ndarray): The reference image (H_ref, W_ref).
        inspected_shape (tuple): Shape of the inspected image (H_insp, W_insp).

    Returns:
        np.ndarray: A larger matrix with the reference image centered.
    """
    H_ref, W_ref = reference.shape
    H_insp, W_insp = inspected_shape

    # Calculate the size of the new matrix
    new_H = H_ref + 2 * H_insp
    new_W = W_ref + 2 * W_insp

    # Create the larger matrix filled with zeros
    centered_matrix = np.zeros((new_H, new_W), dtype=reference.dtype)

    # Calculate the start and end indices to place the reference image at the center
    start_y = H_insp
    start_x = W_insp
    end_y = start_y + H_ref
    end_x = start_x + W_ref

    # Place the reference image at the center
    centered_matrix[start_y:end_y, start_x:end_x] = reference

    # Verify that the first pixel of the inspected image aligns with the corresponding location in the padded reference
    assert centered_matrix[start_y, start_x] == reference[0, 0], (
        f"Alignment check failed: centered_matrix[{start_y}, {start_x}] != reference[0, 0]"
    )

    return centered_matrix


def align_images(reference, inspected):
    """
    Align the inspected image to the reference image using normalized cross-correlation.

    Parameters:
        reference (np.ndarray): The reference image (H, W).
        inspected (np.ndarray): The inspected image (H, W).

    Returns:
        np.ndarray: The aligned inspected image (padded and shifted to match the reference dimensions).
        tuple: The calculated shift (y_shift, x_shift).
    """
    # Center the reference image in a larger matrix
    centered_reference = center_reference(reference, inspected.shape)

    # Perform normalized cross-correlation
    result = cv2.matchTemplate(centered_reference, inspected, method=cv2.TM_CCORR_NORMED)

    # Find the location of the highest correlation
    _, _, _, max_loc = cv2.minMaxLoc(result)
    y_shift, x_shift = max_loc

    # Adjust shifts to account for the centering offset
    y_shift -= inspected.shape[0]
    x_shift -= inspected.shape[1]

    # Align the inspected image by padding and shifting
    aligned_image = pad_and_shift_image(inspected, reference.shape, x_shift, y_shift)

    # Debugging Check: Verify alignment by comparing pixels
    aligned_check_y = max(0, -y_shift)
    aligned_check_x = max(0, -x_shift)

    ref_pixel = reference[0, 0]
    aligned_pixel = aligned_image[aligned_check_y, aligned_check_x] if 0 <= aligned_check_y < aligned_image.shape[0] and 0 <= aligned_check_x < aligned_image.shape[1] else None

    # Print debugging information
    print("\n--- Debugging Information ---")
    print(f"Reference Image Shape: {reference.shape}")
    print(f"Inspected Image Shape: {inspected.shape}")
    print(f"Centered Reference Shape: {centered_reference.shape}")
    print(f"Result Shape: {result.shape}")
    print(f"Calculated Shift: y_shift={y_shift}, x_shift={x_shift}")
    print(f"Aligned Check Coordinates: ({aligned_check_y}, {aligned_check_x})")
    print(f"Reference Pixel Value: {ref_pixel}")
    print(f"Aligned Pixel Value: {aligned_pixel}")
    print("-----------------------------\n")

    assert aligned_pixel == ref_pixel, (
        f"Alignment verification failed: "
        f"Aligned pixel ({aligned_check_y}, {aligned_check_x}) = {aligned_pixel} != "
        f"Reference pixel ({0}, {0}) = {ref_pixel}."
    )

    return aligned_image, (y_shift, x_shift)

if __name__ == "__main__":
    # Load the images
    reference_path = "data/defective_examples/case1_reference_image.tif"
    inspected_path = "data/defective_examples/case1_inspected_image.tif"

    reference = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    inspected = cv2.imread(inspected_path, cv2.IMREAD_GRAYSCALE)

    if reference is None or inspected is None:
        print("Error: Could not load one or both images.")
        sys.exit(1)

    # Align the inspected image to the reference image
    aligned_image, shift_values = align_images(reference, inspected)

    # Visualize results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Reference Image")
    plt.imshow(reference, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Inspected Image")
    plt.imshow(inspected, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(f"Aligned Inspected Image\n(Shift={shift_values})")
    plt.imshow(aligned_image, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Output debugging
    print("Calculated Shift:")
    print(f"y_shift = {shift_values[0]}, x_shift = {shift_values[1]}")