import matplotlib.pyplot as plt
import numpy as np
from filtered_fft_inspected import compute_power_spectrum  # Import the function


def visualize_original_images(reference_image, inspected_image):
    """
    Visualizes the original reference and inspected images side by side.

    Args:
        reference_image (numpy.ndarray): Reference image.
        inspected_image (numpy.ndarray): Inspected image.
    """
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original Reference Image")
    plt.imshow(reference_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Original Inspected Image")
    plt.imshow(inspected_image, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def visualize_aligned_images(reference_image, inspected_image):
    """
    Visualizes the aligned reference and inspected images side by side.

    Args:
        reference_image (numpy.ndarray): Reference image.
        inspected_image (numpy.ndarray): Inspected image.
    """
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Aligned Reference Image")
    plt.imshow(reference_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Aligned Inspected Image")
    plt.imshow(inspected_image, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def visualize_fourier_spectra(power_template, power_test):
    """
    Visualizes the Fourier spectra of the reference and inspected images.

    Args:
        power_template (numpy.ndarray): Power spectrum of the reference image.
        power_test (numpy.ndarray): Power spectrum of the inspected image.
    """
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Reference Power Spectrum (Log Scale)")
    plt.imshow(np.log(1 + power_template), cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Inspected Power Spectrum (Log Scale)")
    plt.imshow(np.log(1 + power_test), cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def visualize_filtered_spectrum(filtered_spectrum):
    """
    Visualizes the filtered Fourier spectrum.

    Args:
        filtered_spectrum (numpy.ndarray): Filtered Fourier spectrum.
    """
    plt.figure(figsize=(6, 6))
    plt.title("Filtered Fourier Spectrum (Log Scale)")
    plt.imshow(np.log(1 + np.abs(filtered_spectrum)), cmap="gray")
    plt.axis("off")
    plt.show()

def visualize_mask(original_image, binary_mask):
    """
    Visualizes the reconstructed image and its binary defect mask.

    Args:
        original_image (numpy.ndarray): Reconstructed spatial domain image.
        binary_mask (numpy.ndarray): Binary mask highlighting defects.
    """
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Reconstructed Image (Defects Highlighted)")
    plt.imshow(original_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Binary Defect Mask")
    plt.imshow(binary_mask, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def visualize_reconstructed_image(reconstructed_image):
    """
    Visualizes the reconstructed spatial domain image.

    Args:
        reconstructed_image (numpy.ndarray): Reconstructed image highlighting defects.
    """
    plt.figure(figsize=(6, 6))
    plt.title("Reconstructed Spatial Domain Image (Defects Highlighted)")
    plt.imshow(reconstructed_image, cmap="gray")
    plt.axis("off")
    plt.show()

def visualize_final_mask(full_mask):
    """
    Visualizes the final binary defect mask.

    Args:
        full_mask (numpy.ndarray): Final binary defect mask.
    """
    plt.figure(figsize=(6, 6))
    plt.title("Final Binary Defect Mask")
    plt.imshow(full_mask, cmap="gray")
    plt.axis("off")
    plt.show()

def visualize_partition_images(partitions_ref, partitions_ins, partition_idx):
    """
    Visualize a specific partition of the reference and inspected images.

    Args:
        partitions_ref (list of numpy.ndarray): List of partitions from the reference image.
        partitions_ins (list of numpy.ndarray): List of partitions from the inspected image.
        partition_idx (int): Index of the partition to visualize.
    """
    ref_partition = partitions_ref[partition_idx]
    ins_partition = partitions_ins[partition_idx]

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title(f"Reference Partition {partition_idx + 1}")
    plt.imshow(ref_partition, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"Inspected Partition {partition_idx + 1}")
    plt.imshow(ins_partition, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def visualize_filtered_spectrum_radial(filtered_spectrum, fft_template, fft_test):
    """
    Visualizes the filtered Fourier spectrum alongside other relevant spectra.
    """
    power_filtered = compute_power_spectrum(filtered_spectrum)
    power_template = compute_power_spectrum(fft_template)
    power_test = compute_power_spectrum(fft_test)

    plt.figure(figsize=(15, 12))

    plt.subplot(3, 1, 1)
    plt.title("Power Spectrum of Template")
    plt.imshow(np.log1p(power_template), cmap="gray")
    plt.axis("off")

    plt.subplot(3, 1, 2)
    plt.title("Power Spectrum of Test")
    plt.imshow(np.log1p(power_test), cmap="gray")
    plt.axis("off")

    plt.subplot(3, 1, 3)
    plt.title("Filtered Power Spectrum (Radial)")
    plt.imshow(np.log1p(power_filtered), cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()