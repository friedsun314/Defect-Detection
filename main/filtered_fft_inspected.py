import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def assign_polar_coordinates(shape):
    """
    Compute polar coordinates (radius, angle) for a 2D array.
    
    Parameters:
        shape (tuple): The shape of the array (height, width).
    
    Returns:
        tuple: radius and angle arrays.
    """
    h, w = shape
    y, x = np.indices((h, w))
    center_y, center_x = h // 2, w // 2
    y -= center_y
    x -= center_x
    radius = np.sqrt(x**2 + y**2)
    angle = (np.arctan2(y, x) + 2 * np.pi) % (2 * np.pi)  # Range: [0, 2π]
    return radius, angle

def compute_power_spectrum(fft_result):
    """
    Compute the power spectrum from an FFT result.
    
    Parameters:
        fft_result (numpy.ndarray): FFT-transformed array.
    
    Returns:
        numpy.ndarray: Power spectrum of the input FFT.
    """
    return np.abs(fft_result) ** 2

def spectrum_subtraction_and_filtering_radial(fft_ref, fft_ins, threshold_ratio=2.0):
    """
    Perform spectrum subtraction and filtering based on the three rules in polar coordinates.
    
    Parameters:
        fft_ref (numpy.ndarray): FFT of the reference image.
        fft_ins (numpy.ndarray): FFT of the inspected image.
        threshold_ratio (float): The ratio to determine significant differences between spectra.
    
    Returns:
        numpy.ndarray: Filtered FFT spectrum.
    """
    # Compute power spectra
    power_ref = compute_power_spectrum(fft_ref)
    power_ins = compute_power_spectrum(fft_ins)
    
    # Assign polar coordinates
    radius, angle = assign_polar_coordinates(fft_ref.shape)
    
    # Create arrays for filtering
    filtered_spectrum = np.zeros_like(fft_ins, dtype=np.complex64)
    
    # Flatten data for easier indexing
    flat_radius = radius.flatten()
    flat_angle = angle.flatten()
    flat_ref = fft_ref.flatten()
    flat_ins = fft_ins.flatten()
    flat_power_ref = power_ref.flatten()
    flat_power_ins = power_ins.flatten()
    
    # Iterate over unique angles
    unique_angles = np.unique(flat_angle)
    for theta in unique_angles:
        # Get indices for this angle
        angle_mask = flat_angle == theta
        radial_indices = flat_radius[angle_mask].argsort()
        
        # Sort values by radius
        sorted_ref = flat_ref[angle_mask][radial_indices]
        sorted_ins = flat_ins[angle_mask][radial_indices]
        sorted_power_ref = flat_power_ref[angle_mask][radial_indices]
        sorted_power_ins = flat_power_ins[angle_mask][radial_indices]
        
        # Apply the three rules
        for i in range(len(sorted_power_ref)):
            current_power_ref = sorted_power_ref[i]
            current_power_ins = sorted_power_ins[i]
            
            # Rule 1: Current power in inspected spectrum must be a local peak
            is_local_peak_ins = (i == 0 or sorted_power_ins[i] > sorted_power_ins[i - 1]) and \
                                (i == len(sorted_power_ins) - 1 or sorted_power_ins[i] > sorted_power_ins[i + 1])
            
            # Rule 2: Current power in reference spectrum must not be a local peak
            is_not_local_peak_ref = (i == 0 or sorted_power_ref[i] <= sorted_power_ref[i - 1]) and \
                                    (i == len(sorted_power_ref) - 1 or sorted_power_ref[i] <= sorted_power_ref[i + 1])
            
            # Rule 3: Power magnitude difference must satisfy threshold ratio
            max_val = max(current_power_ref, current_power_ins)
            min_val = min(current_power_ref, current_power_ins)
            satisfies_ratio = max_val / (min_val + 1e-8) > threshold_ratio
            
            # Combine rules
            if is_local_peak_ins and is_not_local_peak_ref and satisfies_ratio:
                # Select the higher-power coefficient
                filtered_spectrum.flat[angle_mask.nonzero()[0][radial_indices[i]]] = \
                    sorted_ins[i] if current_power_ins > current_power_ref else sorted_ref[i]
    
    return filtered_spectrum