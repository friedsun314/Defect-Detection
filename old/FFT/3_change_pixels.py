import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import cv2


def compute_fast_fourier_transform(image):
    """
    Compute the Fast Fourier Transform (FFT) of a 2D image using numpy.
    """
    return np.fft.fft2(image)


def power_spectrum(fft_result):
    """
    Compute the power spectrum of the Fourier Transform.
    """
    return np.abs(np.fft.fftshift(fft_result))**2


def inverse_fft(fft_result):
    """
    Compute the inverse FFT of a 2D Fourier Transform result.
    """
    return np.fft.ifft2(fft_result).real


def visualize_fft_difference(original_image, modified_image):
    """
    Visualize the FFT of the original image, the modified image, their difference,
    and the IFT of the difference.
    """
    original_fft = compute_fast_fourier_transform(original_image)
    modified_fft = compute_fast_fourier_transform(modified_image)

    original_spectrum = power_spectrum(original_fft)
    modified_spectrum = power_spectrum(modified_fft)

    # Compute the difference in the FFT (complex values)
    fft_difference = modified_fft - original_fft
    difference_spectrum = power_spectrum(fft_difference)

    # Compute the inverse FFT of the difference
    inverse_difference = inverse_fft(fft_difference)

    plt.figure(figsize=(15, 15))

    # Original image and its FFT
    plt.subplot(3, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_image, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 3, 2)
    plt.title("Original Power Spectrum")
    plt.imshow(np.log(original_spectrum + 1), cmap='gray')
    plt.axis('off')

    # Modified image and its FFT
    plt.subplot(3, 3, 4)
    plt.title("Modified Image")
    plt.imshow(modified_image, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 3, 5)
    plt.title("Modified Power Spectrum")
    plt.imshow(np.log(modified_spectrum + 1), cmap='gray')
    plt.axis('off')

    # FFT difference and its inverse FFT
    plt.subplot(3, 3, 8)
    plt.title("Difference Power Spectrum")
    plt.imshow(np.log(difference_spectrum + 1), cmap='gray')
    plt.axis('off')

    plt.subplot(3, 3, 9)
    plt.title("Inverse FFT of Difference")
    plt.imshow(inverse_difference, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def interactive_fft_difference(image_path):
    """
    Provide an interactive interface for marking a region, turning it white,
    and visualizing the difference in FFT.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found. Please check the path.")

    modified_image = image.copy()
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_title("Drag to select a region. Close the window when done.")
    ax.imshow(image, cmap='gray')

    selected_region = [None]  # Use a mutable object to capture the selected region

    def onselect(eclick, erelease):
        """
        Capture the selected region's coordinates and modify the image.
        """
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        selected_region[0] = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

        # Turn the selected region white in the modified image
        x1, y1, x2, y2 = selected_region[0]
        modified_image[y1:y2, x1:x2] = 255

        # Show the rectangle on the image
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', facecolor='none', lw=2))
        fig.canvas.draw()

        print(f"Selected region: {selected_region[0]}")

    # Set up the rectangle selector
    rect_selector = RectangleSelector(ax, onselect, interactive=True, useblit=True)

    plt.show()

    # After marking, visualize the FFT difference
    if selected_region[0] is not None:
        visualize_fft_difference(image, modified_image)
    else:
        print("No region selected. Exiting.")


if __name__ == "__main__":
    # Update the image path below
    image_path = "data/defective_examples/case1_inspected_image.tif"  # Replace with your image path
    interactive_fft_difference(image_path)