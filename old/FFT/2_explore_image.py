import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

def compute_fast_fourier_transform(image):
    """
    Compute the Fast Fourier Transform (FFT) of a 2D image using numpy.
    """
    return np.fft.fft2(image)

def power_spectrum(fft_result):
    """
    Compute the power spectrum of the Fourier Transform.
    """
    shifted_fft = np.fft.fftshift(fft_result)  # Shift the zero frequency component to the center
    return np.log(np.abs(shifted_fft)**2 + 1)  # Compute the power spectrum in log scale

def visualize_fft(image, cropped_region=None):
    """
    Visualize the FFT power spectrum of a 2D image or a cropped region.
    """
    if cropped_region is not None:
        x1, y1, x2, y2 = cropped_region
        cropped_image = image[y1:y2, x1:x2]
    else:
        cropped_image = image

    fft_result = compute_fast_fourier_transform(cropped_image)
    spectrum = power_spectrum(fft_result)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Cropped Image")
    plt.imshow(cropped_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("FFT Power Spectrum")
    plt.imshow(spectrum, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def interactive_fft(image_path):
    """
    Provide an interactive interface for cropping an image and visualizing FFT.
    """
    from matplotlib.widgets import RectangleSelector  # Ensures compatibility with all setups

    image = plt.imread(image_path)
    if len(image.shape) > 2:
        image = np.mean(image, axis=2)  # Convert to grayscale if it's a color image

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_title("Drag to select a region, then close the window.")
    ax.imshow(image, cmap='gray')

    selected_region = [None]  # Use a mutable object to capture the selected region

    def onselect(eclick, erelease):
        """
        Capture the selected region's coordinates.
        """
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        selected_region[0] = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        print(f"Selected region: {selected_region[0]}")

    # Set up the rectangle selector
    rect_selector = RectangleSelector(ax, onselect, interactive=True)

    plt.show()

    # After cropping, visualize the FFT
    if selected_region[0] is not None:
        visualize_fft(image, selected_region[0])
    else:
        print("No region selected. Exiting.")

if __name__ == "__main__":
    # Update the image path below
    image_path = "data/WhatsApp Image 2025-01-18 at 15.46.30.jpeg"  # Replace with your image path
    interactive_fft(image_path)