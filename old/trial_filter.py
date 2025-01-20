import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import zoom

def normalize_spectrum(fft):
    """
    Normalizes the FFT spectrum to have the same overall energy.
    
    Parameters:
        fft (numpy.ndarray): The FFT to normalize.
        
    Returns:
        numpy.ndarray: The normalized FFT.
    """
    energy = np.sum(np.abs(fft) ** 2)
    return fft / np.sqrt(energy)

def upscale_image(image, upscale_factor):
    """
    Upscale the input image by the given factor using bicubic interpolation.
    
    Parameters:
        image (numpy.ndarray): Input image.
        upscale_factor (int): Factor by which to upscale the image.
        
    Returns:
        numpy.ndarray: Upscaled image.
    """
    return zoom(image, upscale_factor, order=3)  # Bicubic interpolation (order=3)

def process_fft(template_fft, test_fft, threshold):
    """
    Processes two shifted FFTs by calculating the absolute difference
    between their power spectra, filtering based on a threshold, and
    selecting the higher-power coefficient for each frequency.
    
    Parameters:
        template_fft (numpy.ndarray): The shifted FFT of the template image.
        test_fft (numpy.ndarray): The shifted FFT of the test image.
        threshold (float): The power threshold for filtering frequencies.
        
    Returns:
        numpy.ndarray: The filtered FFT coefficients.
    """
    # Compute power spectra
    template_power = np.abs(template_fft) ** 2
    test_power = np.abs(test_fft) ** 2
    
    # Calculate the absolute difference
    power_difference = np.abs(template_power - test_power)
    
    # Filter frequencies based on the threshold
    mask = power_difference >= threshold
    
    # Select the higher-power coefficients
    result_fft = np.where(template_power > test_power, template_fft, test_fft)
    
    # Apply the mask to keep only frequencies above the threshold
    result_fft[~mask] = 0
    
    return result_fft

def compute_ift(diff_fft):
    """
    Computes the inverse FFT from a given FFT.
    
    Parameters:
        diff_fft (numpy.ndarray): The FFT difference.
        
    Returns:
        numpy.ndarray: The spatial domain image from the IFT.
    """
    return np.abs(np.fft.ifft2(np.fft.ifftshift(diff_fft)))

def main():
    # Paths to images
    reference_image_path = "output/partitions/reference_partition_4.png"
    test_image_path = "output/partitions/inspected_partition_4.png"
    
    # Load images as grayscale
    reference_image = np.array(Image.open(reference_image_path).convert('L'))
    test_image = np.array(Image.open(test_image_path).convert('L'))
    
    # Upscale images to enhance resolution
    upscale_factor = 1  # Upscale each pixel to 3x3 grid
    reference_image = upscale_image(reference_image, upscale_factor)
    test_image = upscale_image(test_image, upscale_factor)
    
    # Perform FFT and shift
    reference_fft = np.fft.fftshift(np.fft.fft2(reference_image))
    test_fft = np.fft.fftshift(np.fft.fft2(test_image))
    
    # Normalize the spectra to have the same overall energy
    # reference_fft = normalize_spectrum(reference_fft)
    # test_fft = normalize_spectrum(test_fft)
    
    # Define threshold
    threshold = 100000000  # Adjust based on the scale of the power spectra
    
    # Process FFTs
    filtered_fft = process_fft(reference_fft, test_fft, threshold)
    
    # Compute differences
    template_minus_test_fft = normalize_spectrum(reference_fft - test_fft)
    test_minus_template_fft = normalize_spectrum(test_fft - reference_fft)
    
    # Compute inverse FFTs for differences
    template_minus_test_ift = compute_ift(template_minus_test_fft)
    test_minus_template_ift = compute_ift(test_minus_template_fft)
    
    # Compute the inverse FFT of the result
    filtered_image = compute_ift(filtered_fft)
    
    # Plot the FFTs and spatial domain images in multiple windows
    
    # Window 1: Original images and FFT power spectra
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.title("Reference Image (Upscaled)")
    plt.imshow(reference_image, cmap='gray')
    plt.colorbar()
    
    plt.subplot(2, 2, 2)
    plt.title("Reference FFT Power Spectrum (Normalized)")
    plt.imshow(np.log(np.abs(reference_fft) + 1), cmap='gray')
    plt.colorbar()
    
    plt.subplot(2, 2, 3)
    plt.title("Test Image (Upscaled)")
    plt.imshow(test_image, cmap='gray')
    plt.colorbar()
    
    plt.subplot(2, 2, 4)
    plt.title("Test FFT Power Spectrum (Normalized)")
    plt.imshow(np.log(np.abs(test_fft) + 1), cmap='gray')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    # Window 2: Filtered FFT and filtered image
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Filtered FFT Power Spectrum")
    plt.imshow(np.log(np.abs(filtered_fft) + 1), cmap='gray')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.title("Filtered Image (IFT of Result FFT)")
    plt.imshow(filtered_image, cmap='gray')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    # Window 3: Differences in FFTs and their IFTs
    plt.figure(figsize=(12, 12))
    
    plt.subplot(2, 2, 1)
    plt.title("Template - Test FFT Power Spectrum")
    plt.imshow(np.log(np.abs(template_minus_test_fft) + 1), cmap='gray')
    plt.colorbar()
    
    plt.subplot(2, 2, 2)
    plt.title("IFT of Template - Test")
    plt.imshow(template_minus_test_ift, cmap='gray')
    plt.colorbar()
    
    plt.subplot(2, 2, 3)
    plt.title("Test - Template FFT Power Spectrum")
    plt.imshow(np.log(np.abs(test_minus_template_fft) + 1), cmap='gray')
    plt.colorbar()
    
    plt.subplot(2, 2, 4)
    plt.title("IFT of Test - Template")
    plt.imshow(test_minus_template_ift, cmap='gray')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()