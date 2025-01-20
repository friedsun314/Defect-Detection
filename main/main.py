import os
import cv2
import numpy as np
from preprocessing import sift_alignment_and_cropping, preprocess_images
from partitioning import partition_image
from fft import compute_fourier_and_power_spectrum
from filtered_fft_inspected import spectrum_subtraction_and_filtering_radial
from ift import inverse_fourier_transform, visualize_reconstructed_image
from threshold_masking import adaptive_thresholding, postprocess_mask
from postprocessing import reconstruct_full_mask
from visualization import (
    visualize_original_images,
    visualize_aligned_images,
    visualize_final_mask,
    visualize_mask,
    visualize_filtered_spectrum_radial,
    visualize_partition_images,
)


def save_partitions(partitions, base_path, prefix):
    """
    Save the partitioned images to disk.

    Args:
        partitions (list of numpy.ndarray): List of partitioned sub-images.
        base_path (str): Directory where the partitions will be saved.
        prefix (str): Prefix for the filenames (e.g., "reference" or "inspected").
    """
    os.makedirs(base_path, exist_ok=True)
    for idx, partition in enumerate(partitions):
        filename = os.path.join(base_path, f"{prefix}_partition_{idx + 1}.png")
        cv2.imwrite(filename, partition)
        print(f"Saved {filename}")


def main(reference_image_path=None, inspected_image_path=None, visualize_steps=True):
    if reference_image_path is None or inspected_image_path is None:
        raise ValueError("Both reference and inspected image paths must be provided.")

    # Step 0: Load the images
    print("Loading images...")
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    inspected_image = cv2.imread(inspected_image_path, cv2.IMREAD_GRAYSCALE)

    if reference_image is None:
        raise FileNotFoundError(f"Reference image not found at {reference_image_path}")
    if inspected_image is None:
        raise FileNotFoundError(f"Inspected image not found at {inspected_image_path}")

    # Step 1: Alignment and Cropping
    print("Aligning and cropping images...")
    aligned_reference, aligned_inspected = sift_alignment_and_cropping(reference_image, inspected_image)

    # Visualize original and aligned images
    if visualize_steps:
        visualize_original_images(reference_image, inspected_image)
        visualize_aligned_images(aligned_reference, aligned_inspected)

    # Step 2: Partitioning - Divide images into overlapping subimages
    print("Partitioning images...")
    partitions = (4, 4)  # Number of partitions (rows, cols)
    overlap = 0  # Overlap fraction
    subimages_ref, positions_ref = partition_image(aligned_reference, partitions, overlap)
    subimages_ins, positions_ins = partition_image(aligned_inspected, partitions, overlap)

    # Save partitions to disk
    save_partitions(subimages_ref, "output/partitions", "reference")
    save_partitions(subimages_ins, "output/partitions", "inspected")

    # Ensure partition positions match
    if positions_ref != positions_ins:
        raise ValueError("Reference and inspected image partitions do not align.")

    # Step 3: Processing each partition
    print("Processing partitions...")
    partition_masks = []
    for idx, (ref_sub, ins_sub) in enumerate(zip(subimages_ref, subimages_ins)):
        print(f"Processing partition {idx + 1}/{len(subimages_ref)}...")

        # Visualize the current partition
        if visualize_steps:
            visualize_partition_images(subimages_ref, subimages_ins, idx)

        # Step 3.1: Compute Fourier Transform and power spectra
        fft_ref, power_ref = compute_fourier_and_power_spectrum(ref_sub)
        fft_ins, power_ins = compute_fourier_and_power_spectrum(ins_sub)

        # Step 3.2: Spectrum subtraction and filtering (radial-based)
        filtered_spectrum = spectrum_subtraction_and_filtering_radial(fft_ref, fft_ins)
        
        # Visualize filtered spectrum
        if visualize_steps:
            visualize_filtered_spectrum_radial(filtered_spectrum, fft_ref, fft_ins)

        # Step 3.3: Inverse Fourier Transform - Reconstruct spatial domain image
        reconstructed_image = inverse_fourier_transform(filtered_spectrum)

        # Visualize reconstructed image
        if visualize_steps:
            visualize_reconstructed_image(reconstructed_image)

        # Step 3.4: Thresholding and segmentation
        binary_mask = adaptive_thresholding(reconstructed_image, threshold_factor=3)
        cleaned_mask = postprocess_mask(binary_mask, kernel_size=3)

        # Visualize mask for the current partition
        if visualize_steps:
            visualize_mask(reconstructed_image, cleaned_mask)

        # Collect the binary mask for the current partition
        partition_masks.append(cleaned_mask)

    # Step 4: Postprocessing - Reconstruct the full mask
    print("Reconstructing the full defect mask...")
    full_mask = reconstruct_full_mask(partition_masks, positions_ins, aligned_inspected.shape)

    # Step 5: Visualization - Display final results
    print("Visualizing final results...")
    visualize_final_mask(full_mask)

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    reference_image_path = "data/defective_examples/case1_reference_image.tif"
    inspected_image_path = "data/defective_examples/case1_inspected_image.tif"
    main(reference_image_path, inspected_image_path)