import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import zoom

def fft_to_polar(image_fft):
    """
    Convert FFT data to polar coordinates by computing magnitude and angles.
    
    Parameters:
        image_fft (numpy.ndarray): FFT data of the image.
        
    Returns:
        tuple: Radii and angles of the FFT in polar coordinates.
    """
    h, w = image_fft.shape
    y, x = np.meshgrid(np.arange(-h//2, h//2), np.arange(-w//2, w//2), indexing='ij')
    radii = np.sqrt(x**2 + y**2)
    angles = np.arctan2(y, x)
    return radii, angles

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

def identify_local_maxima(values):
    """
    Identify indices of local maxima in a 1D array.
    
    Parameters:
        values (numpy.ndarray): Array of values to analyze.
        
    Returns:
        list: Indices of local maxima.
    """
    maxima_indices = []
    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            maxima_indices.append(i)
    return maxima_indices

def process_rays(template_fft, test_fft, num_rays, measure="normalized_mean"):
    """
    Process FFT slices along angular rays and calculate metrics (normalized mean or median).
    
    Parameters:
        template_fft (numpy.ndarray): FFT of the template image.
        test_fft (numpy.ndarray): FFT of the test image.
        num_rays (int): Number of angular rays to analyze.
        measure (str): Metric to calculate for each slice ("normalized_mean" or "median").
        
    Returns:
        tuple: Filtered FFT of the test image, local maxima indices, template metrics.
    """
    _, angles = fft_to_polar(template_fft)
    angle_bins = np.linspace(-np.pi, np.pi, num_rays + 1)

    template_metrics = []
    ray_masks = []

    for i in range(num_rays):
        # Create mask for the current slice
        mask = (angles >= angle_bins[i]) & (angles < angle_bins[i + 1])
        ray_masks.append(mask)
        
        # Compute the selected metric
        if measure == "normalized_mean":
            # Area-normalized mean: sum of pixel values divided by the number of pixels
            slice_area = np.sum(mask)
            slice_value = np.sum(np.abs(template_fft[mask])) / slice_area if slice_area > 0 else 0
        elif measure == "median":
            # Median of the slice
            slice_value = np.median(np.abs(template_fft[mask])) if np.any(mask) else 0
        else:
            raise ValueError("Invalid measure: choose 'normalized_mean' or 'median'")
        
        template_metrics.append(slice_value)

    # Identify local maxima in the template metrics
    local_maxima_indices = identify_local_maxima(template_metrics)

    # Zero out the corresponding pixels in the test FFT
    filtered_fft = np.zeros_like(test_fft, dtype=test_fft.dtype)
    for idx in local_maxima_indices:
        filtered_fft[ray_masks[idx]] = test_fft[ray_masks[idx]]

    return filtered_fft, local_maxima_indices, template_metrics



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
    
    # Store original FFTs for comparison
    original_reference_fft = np.abs(reference_fft)
    original_test_fft = np.abs(test_fft)
    
    # Number of angular rays to analyze
    num_rays = 18  # One ray per degree
    
    # Process rays and get the filtered FFT
    filtered_fft, local_maxima_indices, template_means = process_rays(reference_fft, test_fft, num_rays)
    
    # Compute the inverse FFT of the filtered result
    filtered_image = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_fft)))
    
    # Visualization

    # Window 1: Original images and their FFTs
    plt.figure(figsize=(18, 12))
    
    plt.subplot(2, 2, 1)
    plt.title("Reference Image (Upscaled)")
    plt.imshow(reference_image, cmap='gray')
    plt.colorbar()
    
    plt.subplot(2, 2, 2)
    plt.title("Reference FFT Power Spectrum")
    plt.imshow(np.log(original_reference_fft + 1), cmap='gray')
    plt.colorbar()
    
    plt.subplot(2, 2, 3)
    plt.title("Test Image (Upscaled)")
    plt.imshow(test_image, cmap='gray')
    plt.colorbar()
    
    plt.subplot(2, 2, 4)
    plt.title("Test FFT Power Spectrum")
    plt.imshow(np.log(original_test_fft + 1), cmap='gray')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

    # Window 2: Ray analysis and filtered FFT
    plt.figure(figsize=(18, 12))
    
    # Plot ray means for template
    plt.subplot(2, 2, 1)
    plt.title("Template Ray Means")
    plt.plot(template_means, label="Template Ray Means")
    plt.scatter(local_maxima_indices, [template_means[i] for i in local_maxima_indices],
                color='red', label="Local Maxima")
    plt.xlabel("Ray Index")
    plt.ylabel("Mean Value")
    plt.legend()
    
    # Plot filtered FFT power spectrum
    plt.subplot(2, 2, 2)
    plt.title("Filtered Test FFT Power Spectrum")
    plt.imshow(np.log(np.abs(filtered_fft) + 1), cmap='gray')
    plt.colorbar()
    
    # Plot filtered image (IFT of filtered FFT)
    plt.subplot(2, 2, 3)
    plt.title("Filtered Image (IFT of Filtered FFT)")
    plt.imshow(filtered_image, cmap='gray')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

    # Debug: Print local maxima indices
    print("Local maxima indices (angles):", local_maxima_indices)

if __name__ == "__main__":
    main()