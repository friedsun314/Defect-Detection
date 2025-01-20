import numpy as np
import matplotlib.pyplot as plt
import cv2


def pad_and_shift_image(inspected, reference_shape, y_shift, x_shift):
    """
    Pad the inspected image to match the reference dimensions and apply the shift.

    Parameters:
        inspected (np.ndarray): The inspected image (H, W).
        reference_shape (tuple): Shape of the reference image (H_ref, W_ref).
        y_shift (int): Vertical shift (positive: down, negative: up).
        x_shift (int): Horizontal shift (positive: right, negative: left).

    Returns:
        np.ndarray: The padded and shifted inspected image.
    """
    H_ref, W_ref = reference_shape
    H_insp, W_insp = inspected.shape

    # Create a zero matrix with the same dimensions as the reference
    padded = np.zeros((H_ref, W_ref), dtype=inspected.dtype)

    # Copy the inspected image into the upper-left corner of the padded image
    padded[:H_insp, :W_insp] = inspected

    # Create an output matrix for the shifted result
    shifted = np.zeros_like(padded)

    # Calculate valid regions for shifting
    start_y = max(0, y_shift)
    start_x = max(0, x_shift)
    end_y = min(H_ref, H_insp + y_shift)
    end_x = min(W_ref, W_insp + x_shift)

    # Calculate the corresponding slice in the padded image
    src_start_y = max(0, -y_shift)
    src_start_x = max(0, -x_shift)
    src_end_y = src_start_y + (end_y - start_y)
    src_end_x = src_start_x + (end_x - start_x)

    # Copy the relevant region from the padded image to the shifted image
    shifted[start_y:end_y, start_x:end_x] = padded[src_start_y:src_end_y, src_start_x:src_end_x]

    return shifted

# --- Testing and Visualization ---
if __name__ == "__main__":
    # Example 1: Rectangle Test
    reference = np.zeros((200, 200), dtype=np.uint8)
    reference[50:150, 50:150] = 255  # White rectangle at the center

    inspected = np.zeros((100, 100), dtype=np.uint8)
    inspected[20:80, 20:80] = 255  # Smaller rectangle

    y_shift, x_shift = 30, -30
    padded_shifted = pad_and_shift_image(inspected, reference.shape, y_shift, x_shift)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Reference Image (Rectangle)")
    plt.imshow(reference, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Inspected Image (Rectangle)")
    plt.imshow(inspected, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(f"Padded and Shifted (Shift={y_shift}, {x_shift})")
    plt.imshow(padded_shifted, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Example 2: Checkerboard with Line
    reference = np.zeros((200, 200), dtype=np.uint8)
    reference[::20, ::20] = 255  # Add a checkerboard pattern
    cv2.line(reference, (0, 100), (200, 100), 255, 3)  # Add a horizontal white line

    inspected = reference[40:140, 40:140]  # Extract a sub-region
    y_shift, x_shift = 60, 40
    padded_shifted = pad_and_shift_image(inspected, reference.shape, y_shift, x_shift)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Reference Image (Checkerboard with Line)")
    plt.imshow(reference, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Inspected Image (Checkerboard with Line)")
    plt.imshow(inspected, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(f"Padded and Shifted (Shift={y_shift}, {x_shift})")
    plt.imshow(padded_shifted, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()