import numpy as np
import matplotlib.pyplot as plt

def compute_fast_fourier_transform(image):
    """
    Compute the Fast Fourier Transform (FFT) of a 2D image using numpy.
    """
    return np.fft.fft2(image)

def magnitude_spectrum(fft_result):
    """
    Compute the magnitude spectrum of the Fourier Transform.
    """
    return np.log(np.abs(np.fft.fftshift(fft_result)) + 1) ** 2

def visualize_fft(image):
    """
    Visualize the FFT magnitude spectrum of a 2D image.
    """
    fft_result = compute_fast_fourier_transform(image)
    spectrum = magnitude_spectrum(fft_result)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("FFT Magnitude Spectrum")
    plt.imshow(spectrum, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def generate_simple_images():
    """
    Generate simple test images for FFT visualization.
    """
    # Checkerboard pattern
    checkerboard = np.zeros((64, 64))
    for i in range(64):
        for j in range(64):
            if (i // 8 + j // 8) % 2 == 0:
                checkerboard[i, j] = 255

    # Gradient image
    gradient = np.linspace(0, 255, 64).reshape(1, -1)
    gradient = np.tile(gradient, (64, 1))

    # Single bright pixel
    single_pixel = np.zeros((64, 64))
    single_pixel[32, 32] = 255

    # Horizontal lines
    horizontal_lines = np.zeros((64, 64))
    horizontal_lines[::8, :] = 255

    # Vertical lines
    vertical_lines = np.zeros((64, 64))
    vertical_lines[:, ::8] = 255

    # Circular pattern
    circular_pattern = np.zeros((64, 64))
    center = (32, 32)
    for x in range(64):
        for y in range(64):
            if (x - center[0])**2 + (y - center[1])**2 <= 15**2:
                circular_pattern[x, y] = 255

    return {
        "checkerboard": checkerboard,
        "gradient": gradient,
        "single_pixel": single_pixel,
        "horizontal_lines": horizontal_lines,
        "vertical_lines": vertical_lines,
        "circular_pattern": circular_pattern,
    }

if __name__ == "__main__":
    # Generate simple images
    simple_images = generate_simple_images()
    
    # Visualize FFT for each generated image
    for name, image in simple_images.items():
        print(f"Visualizing FFT for {name} image...")
        visualize_fft(image)