import numpy as np
import matplotlib.pyplot as plt


def generate_structured_spectrum(shape, pattern_type="stripes", frequency=10):
    """
    Generates a structured Fourier spectrum with known patterns.

    Args:
        shape (tuple): Shape of the spectrum (height, width).
        pattern_type (str): Type of pattern ('stripes', 'checkerboard', 'circles', 'diagonal').
        frequency (int): Frequency of the pattern.

    Returns:
        numpy.ndarray: Structured Fourier spectrum (complex coefficients).
    """
    h, w = shape
    spectrum = np.zeros((h, w), dtype=np.complex64)

    if pattern_type == "stripes":
        for y in range(0, h, frequency):
            spectrum[y, :] = 1 + 1j  # High values along rows
    elif pattern_type == "checkerboard":
        for y in range(0, h, frequency):
            for x in range(0, w, frequency):
                if (y // frequency + x // frequency) % 2 == 0:
                    spectrum[y:y + frequency // 2, x:x + frequency // 2] = 1 + 1j
    elif pattern_type == "circles":
        center = (h // 2, w // 2)
        for r in range(0, min(h, w) // 2, frequency):
            y, x = np.ogrid[:h, :w]
            mask = (x - center[1])**2 + (y - center[0])**2 < r**2
            spectrum[mask] += 1 + 1j
    elif pattern_type == "diagonal":
        for d in range(0, h, frequency):
            for i in range(min(h - d, w)):
                spectrum[d + i, i] = 1 + 1j
                spectrum[i, d + i] = 1 + 1j

    return spectrum


def inverse_fourier_transform(filtered_spectrum):
    """
    Reconstructs the spatial domain image from the filtered Fourier spectrum.

    Args:
        filtered_spectrum (numpy.ndarray): Filtered Fourier spectrum (complex coefficients).

    Returns:
        numpy.ndarray: Reconstructed spatial domain image highlighting potential defects.
    """
    # Shift the zero frequency component back to the corners
    unshifted_spectrum = np.fft.ifftshift(filtered_spectrum)
    # Perform the inverse Fourier Transform
    reconstructed_image = np.fft.ifft2(unshifted_spectrum)
    # Take the magnitude of the result (real-valued image)
    reconstructed_magnitude = np.abs(reconstructed_image)

    return reconstructed_magnitude


def visualize_reconstructed_image(reconstructed_image):
    """
    Visualizes the reconstructed spatial domain image.

    Args:
        reconstructed_image (numpy.ndarray): Reconstructed image highlighting defects.
    """
    plt.figure(figsize=(6, 6))
    plt.title("Reconstructed Spatial Domain Image")
    plt.imshow(reconstructed_image, cmap="gray")
    plt.axis("off")
    plt.show()


def run_test_case(template_spectrum, test_spectrum, threshold_ratio=1.5, title=""):
    """
    Runs the filtering process on a test case and visualizes results.

    Args:
        template_spectrum (numpy.ndarray): Fourier spectrum of the template (complex coefficients).
        test_spectrum (numpy.ndarray): Fourier spectrum of the test (complex coefficients).
        threshold_ratio (float): Threshold for significant differences.
        title (str): Title for the test case visualization.
    """
    # Subtract the Fourier spectra to highlight defects
    filtered_spectrum = test_spectrum - template_spectrum
    magnitude_template = np.abs(template_spectrum)
    magnitude_test = np.abs(test_spectrum)
    reconstructed_image = inverse_fourier_transform(filtered_spectrum)


if __name__ == "__main__":
    h, w = 256, 256  # Shape of the spectra

    # Example 1: Horizontal Stripes
    template = generate_structured_spectrum((h, w), pattern_type="stripes", frequency=10)
    test = template.copy()
    test[128, 128] += 10 + 5j  # Add defect
    run_test_case(template, test, title="Example 1: Horizontal Stripes")

    # Example 2: Checkerboard Pattern
    template = generate_structured_spectrum((h, w), pattern_type="checkerboard", frequency=20)
    test = template.copy()
    test[100, 150] += 15 + 3j  # Add defect
    run_test_case(template, test, title="Example 2: Checkerboard Pattern")

    # Example 3: Concentric Circles
    template = generate_structured_spectrum((h, w), pattern_type="circles", frequency=15)
    test = template.copy()
    test[150, 100] += 20 + 7j  # Add defect
    run_test_case(template, test, title="Example 3: Concentric Circles")

    # Example 4: Diagonal Lines
    template = generate_structured_spectrum((h, w), pattern_type="diagonal", frequency=10)
    test = template.copy()
    test[200, 50] += 12 + 4j  # Add defect
    run_test_case(template, test, title="Example 4: Diagonal Lines")

import numpy as np
import matplotlib.pyplot as plt


def generate_structured_spectrum(shape, pattern_type="stripes", frequency=10):
    """
    Generates a structured Fourier spectrum with known patterns.

    Args:
        shape (tuple): Shape of the spectrum (height, width).
        pattern_type (str): Type of pattern ('stripes', 'checkerboard', 'circles', 'diagonal').
        frequency (int): Frequency of the pattern.

    Returns:
        numpy.ndarray: Structured Fourier spectrum (complex coefficients).
    """
    h, w = shape
    spectrum = np.zeros((h, w), dtype=np.complex64)

    if pattern_type == "stripes":
        for y in range(0, h, frequency):
            spectrum[y, :] = 1 + 1j  # High values along rows
    elif pattern_type == "checkerboard":
        for y in range(0, h, frequency):
            for x in range(0, w, frequency):
                if (y // frequency + x // frequency) % 2 == 0:
                    spectrum[y:y + frequency // 2, x:x + frequency // 2] = 1 + 1j
    elif pattern_type == "circles":
        center = (h // 2, w // 2)
        for r in range(0, min(h, w) // 2, frequency):
            y, x = np.ogrid[:h, :w]
            mask = (x - center[1])**2 + (y - center[0])**2 < r**2
            spectrum[mask] += 1 + 1j
    elif pattern_type == "diagonal":
        for d in range(0, h, frequency):
            for i in range(min(h - d, w)):
                spectrum[d + i, i] = 1 + 1j
                spectrum[i, d + i] = 1 + 1j

    return spectrum


def inverse_fourier_transform(filtered_spectrum):
    """
    Reconstructs the spatial domain image from the filtered Fourier spectrum.

    Args:
        filtered_spectrum (numpy.ndarray): Filtered Fourier spectrum (complex coefficients).

    Returns:
        numpy.ndarray: Reconstructed spatial domain image highlighting potential defects.
    """
    # Shift the zero frequency component back to the corners
    unshifted_spectrum = np.fft.ifftshift(filtered_spectrum)
    # Perform the inverse Fourier Transform
    reconstructed_image = np.fft.ifft2(unshifted_spectrum)
    # Take the magnitude of the result (real-valued image)
    reconstructed_magnitude = np.abs(reconstructed_image)

    return reconstructed_magnitude


def visualize_reconstructed_image(reconstructed_image):
    """
    Visualizes the reconstructed spatial domain image.

    Args:
        reconstructed_image (numpy.ndarray): Reconstructed image highlighting defects.
    """
    plt.figure(figsize=(6, 6))
    plt.title("Reconstructed Spatial Domain Image")
    plt.imshow(reconstructed_image, cmap="gray")
    plt.axis("off")
    plt.show()


def run_test_case(template_spectrum, test_spectrum, threshold_ratio=1.5, title=""):
    """
    Runs the filtering process on a test case and visualizes results.

    Args:
        template_spectrum (numpy.ndarray): Fourier spectrum of the template (complex coefficients).
        test_spectrum (numpy.ndarray): Fourier spectrum of the test (complex coefficients).
        threshold_ratio (float): Threshold for significant differences.
        title (str): Title for the test case visualization.
    """
    # Subtract the Fourier spectra to highlight defects
    filtered_spectrum = test_spectrum - template_spectrum
    magnitude_template = np.abs(template_spectrum)
    magnitude_test = np.abs(test_spectrum)
    reconstructed_image = inverse_fourier_transform(filtered_spectrum)


if __name__ == "__main__":
    h, w = 256, 256  # Shape of the spectra

    # Example 1: Horizontal Stripes
    template = generate_structured_spectrum((h, w), pattern_type="stripes", frequency=10)
    test = template.copy()
    test[128, 128] += 10 + 5j  # Add defect
    run_test_case(template, test, title="Example 1: Horizontal Stripes")

    # Example 2: Checkerboard Pattern
    template = generate_structured_spectrum((h, w), pattern_type="checkerboard", frequency=20)
    test = template.copy()
    test[100, 150] += 15 + 3j  # Add defect
    run_test_case(template, test, title="Example 2: Checkerboard Pattern")

    # Example 3: Concentric Circles
    template = generate_structured_spectrum((h, w), pattern_type="circles", frequency=15)
    test = template.copy()
    test[150, 100] += 20 + 7j  # Add defect
    run_test_case(template, test, title="Example 3: Concentric Circles")

    # Example 4: Diagonal Lines
    template = generate_structured_spectrum((h, w), pattern_type="diagonal", frequency=10)
    test = template.copy()
    test[200, 50] += 12 + 4j  # Add defect
    run_test_case(template, test, title="Example 4: Diagonal Lines")

    # Example 5: c2 from the paper
    import cv2

    path = "data/WhatsApp Image 2025-01-18 at 15.30.27.jpeg"
    # Load the image in grayscale
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Image not found at {path}")

    # Compute the Fourier Transform of the image
    fft_image = np.fft.fft2(image)
    shifted_fft = np.fft.fftshift(fft_image)  # Shift zero frequency to the center

    # Use the inverse_fourier_transform to reconstruct
    reconstructed_image = inverse_fourier_transform(shifted_fft)

    # Visualize the reconstructed image
    visualize_reconstructed_image(reconstructed_image)