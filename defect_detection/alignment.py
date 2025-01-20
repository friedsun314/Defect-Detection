import numpy as np
from scipy.signal import correlate2d
from scipy.ndimage import shift
import matplotlib.pyplot as plt

def cross_correlation_align(ref_image, inspected_image, debug=False):
    """
    Align the inspected image to the reference image using cross-correlation.

    Parameters:
        ref_image (np.ndarray): The reference image (grayscale).
        inspected_image (np.ndarray): The inspected image (grayscale).
        debug (bool): If True, outputs debugging information and visualizations.

    Returns:
        aligned_image (np.ndarray): The aligned inspected image.
        shift_values (tuple): (y_shift, x_shift) applied to align the images.
    """
    # Compute cross-correlation
    correlation = correlate2d(ref_image, inspected_image, mode='full', boundary='fill')

    # Debug: Visualize the correlation map
    if debug:
        plt.figure(figsize=(10, 5))
        plt.title("Cross-Correlation Map")
        plt.imshow(correlation, cmap='viridis')
        plt.colorbar()
        plt.show()

    # Find the peak correlation (indicating best alignment)
    peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)

    # Calculate shift values relative to the center of the correlation matrix
    corr_center_y, corr_center_x = np.array(correlation.shape) // 2
    y_shift = peak_y - corr_center_y
    x_shift = peak_x - corr_center_x

    # Debug: Print peak location and shift values
    if debug:
        print(f"Peak Correlation Location: (y={peak_y}, x={peak_x})")
        print(f"Correlation Center: (y={corr_center_y}, x={corr_center_x})")
        print(f"Calculated Shift: (y_shift={y_shift}, x_shift={x_shift})")

    # Apply the shift to the inspected image
    aligned_image = shift(inspected_image, shift=(-y_shift, -x_shift), mode='nearest')

    # Debug: Visualize the aligned image
    if debug:
        plt.figure(figsize=(10, 5))
        plt.title("Aligned Image")
        plt.imshow(aligned_image, cmap='gray')
        plt.show()

    return aligned_image, (y_shift, x_shift)