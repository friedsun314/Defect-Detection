import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_fourier_and_power_spectrum(image):
    """
    Computes the Fourier Transform and Power Spectrum of an input image.

    Args:
        image (numpy.ndarray): Input grayscale image.

    Returns:
        numpy.ndarray: Fourier transform of the image.
        numpy.ndarray: Power spectrum of the image.
    """
    # Compute the 2D Fourier Transform
    fourier_transform = np.fft.fft2(image)
    # Shift the zero frequency component to the center
    shifted_fourier = np.fft.fftshift(fourier_transform)
    # Compute the power spectrum (magnitude squared)
    power_spectrum = np.abs(shifted_fourier) ** 2

    return shifted_fourier, power_spectrum

def visualize_fourier_and_power(reference_image, inspected_image):
    """
    Computes and visualizes the Fourier Transform and Power Spectrum for both images.

    Args:
        reference_image (numpy.ndarray): Reference image.
        inspected_image (numpy.ndarray): Inspected image.
    """
    # Compute Fourier transforms and power spectra
    fourier_ref, power_ref = compute_fourier_and_power_spectrum(reference_image)
    fourier_ins, power_ins = compute_fourier_and_power_spectrum(inspected_image)

    # Visualize results
    plt.figure(figsize=(12, 8))

    # Reference image visualization
    plt.subplot(2, 3, 1)
    plt.title("Reference Image")
    plt.imshow(reference_image, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.title("Reference Fourier Transform (Log Scale)")
    plt.imshow(np.log(1 + np.abs(fourier_ref)), cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.title("Reference Power Spectrum (Log Scale)")
    plt.imshow(np.log(1 + power_ref), cmap="gray")
    plt.axis("off")

    # Inspected image visualization
    plt.subplot(2, 3, 4)
    plt.title("Inspected Image")
    plt.imshow(inspected_image, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.title("Inspected Fourier Transform (Log Scale)")
    plt.imshow(np.log(1 + np.abs(fourier_ins)), cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.title("Inspected Power Spectrum (Log Scale)")
    plt.imshow(np.log(1 + power_ins), cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    return fourier_ref, power_ref, fourier_ins, power_ins

if __name__ == "__main__":
    from preprocessing import sift_alignment_and_cropping  # Import the preprocessing function

    # Paths to original images
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

    # Perform alignment and cropping using the preprocessing function
    try:
        cropped_reference, cropped_inspected = sift_alignment_and_cropping(reference_image, inspected_image)
    except ValueError as e:
        print(f"Alignment failed: {e}")
        exit()

    # Compute and visualize Fourier Transform and Power Spectrum
    fourier_ref, power_ref, fourier_ins, power_ins = visualize_fourier_and_power(cropped_reference, cropped_inspected)