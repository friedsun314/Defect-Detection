import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_power_spectrum(fft_result):
    """
    Compute the power spectrum of the Fourier Transform.
    """
    return np.abs(fft_result) ** 2

def detect_primary_frequencies(fft_reference, threshold_ratio=0.1):
    """
    Detect primary frequencies in the reference image's Fourier spectrum.

    Args:
        fft_reference (numpy.ndarray): Fourier Transform of the reference image.
        threshold_ratio (float): Ratio of the maximum power spectrum to consider as primary frequencies.

    Returns:
        numpy.ndarray: Binary mask of primary frequencies.
    """
    power_spectrum = compute_power_spectrum(fft_reference)
    max_power = np.max(power_spectrum)
    threshold = max_power * threshold_ratio

    # Create a binary mask for frequencies above the threshold
    primary_frequency_mask = power_spectrum > threshold
    return primary_frequency_mask

def apply_frequency_mask(fft_test, frequency_mask):
    """
    Apply the frequency mask to the Fourier Transform of the inspected image.

    Args:
        fft_test (numpy.ndarray): Fourier Transform of the inspected image.
        frequency_mask (numpy.ndarray): Binary mask of primary frequencies.

    Returns:
        numpy.ndarray: Filtered Fourier spectrum of the inspected image.
    """
    return fft_test * frequency_mask

def inverse_fourier_transform(filtered_spectrum):
    """
    Reconstruct the spatial domain image from the filtered Fourier spectrum.

    Args:
        filtered_spectrum (numpy.ndarray): Filtered Fourier spectrum.

    Returns:
        numpy.ndarray: Reconstructed spatial domain image.
    """
    shifted_spectrum = np.fft.ifftshift(filtered_spectrum)
    reconstructed_image = np.fft.ifft2(shifted_spectrum)
    return np.abs(reconstructed_image)

def visualize_images(reference_image, defected_image, filtered_image):
    """
    Visualize the reference, defected, and filtered images.

    Args:
        reference_image (numpy.ndarray): Reference image.
        defected_image (numpy.ndarray): Defected image.
        filtered_image (numpy.ndarray): Reconstructed image after applying the frequency mask.
    """
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title("Reference Image")
    plt.imshow(reference_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Defected Image")
    plt.imshow(defected_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Filtered Image")
    plt.imshow(filtered_image, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load the reference image
    reference_image_path = "data/defective_examples/case1_reference_image.tif"
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    if reference_image is None:
        raise FileNotFoundError(f"Reference image not found at {reference_image_path}")

    # Add a random defect to the reference image
    defected_image = reference_image.copy()
    defected_image[100:104, 100:104] = 255  # Add a small defect

    # Compute FFTs
    fft_reference = np.fft.fft2(reference_image)
    fft_test = np.fft.fft2(defected_image)

    # Detect primary frequencies in the reference image
    primary_frequency_mask = detect_primary_frequencies(fft_reference, threshold_ratio=0.1)

    # Apply the mask to the inspected image
    filtered_fft_test = apply_frequency_mask(fft_test, primary_frequency_mask)

    # Reconstruct the spatial domain image from the filtered spectrum
    reconstructed_image = inverse_fourier_transform(filtered_fft_test)

    # Visualize results
    visualize_images(reference_image, defected_image, reconstructed_image)

    # Optional: Save the frequency mask visualization
    plt.figure(figsize=(6, 6))
    plt.title("Primary Frequency Mask")
    plt.imshow(primary_frequency_mask, cmap="gray")
    plt.axis("off")
    plt.show()