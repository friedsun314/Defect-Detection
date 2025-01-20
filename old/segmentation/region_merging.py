import numpy as np
import matplotlib.pyplot as plt
import cv2
from region_growing import region_growing_with_full_coverage


def preprocess_image(image, method1="gaussian", kernel_size1=5, method2="median", kernel_size2=3):
    """
    Preprocess the image by applying two smoothing steps to the background.

    :param image: Grayscale input image (H x W).
    :param method1: First smoothing method ('gaussian' or 'median').
    :param kernel_size1: Kernel size for the first smoothing step.
    :param method2: Second smoothing method ('gaussian' or 'median').
    :param kernel_size2: Kernel size for the second smoothing step.
    :return: Smoothed image.
    """
    # First smoothing step
    if method1 == "gaussian":
        smoothed = cv2.GaussianBlur(image, (kernel_size1, kernel_size1), 0)
    elif method1 == "median":
        smoothed = cv2.medianBlur(image, kernel_size1)
    else:
        raise ValueError(f"Unsupported method: {method1}. Use 'gaussian' or 'median'.")

    # Second smoothing step
    if method2 == "gaussian":
        smoothed = cv2.GaussianBlur(smoothed, (kernel_size2, kernel_size2), 0)
    elif method2 == "median":
        smoothed = cv2.medianBlur(smoothed, kernel_size2)
    else:
        raise ValueError(f"Unsupported method: {method2}. Use 'gaussian' or 'median'.")

    return smoothed


def merge_similar_regions(image, regions, similarity_threshold=10.0):
    """
    Merge regions that have similar mean intensity values.

    :param image: Grayscale image (H x W).
    :param regions: List of regions, where each region is a set of (x, y) pixel coordinates.
    :param similarity_threshold: Merge regions if their mean intensities differ by less than this value.
    :return: A new list of merged regions.
    """
    # Compute mean intensities for all regions
    region_means = []
    for region in regions:
        values = [image[y, x] for x, y in region]
        region_means.append(np.mean(values))

    merged = True
    while merged:
        merged = False
        new_regions = []
        new_means = []
        skip_indices = set()

        for i in range(len(regions)):
            if i in skip_indices:
                continue

            current_region = regions[i]
            current_mean = region_means[i]

            for j in range(i + 1, len(regions)):
                if j in skip_indices:
                    continue

                other_region = regions[j]
                other_mean = region_means[j]

                # Check similarity
                if abs(current_mean - other_mean) <= similarity_threshold:
                    # Merge regions
                    current_region = current_region.union(other_region)
                    current_mean = np.mean([image[y, x] for x, y in current_region])
                    skip_indices.add(j)
                    merged = True

            new_regions.append(current_region)
            new_means.append(current_mean)

        regions = new_regions
        region_means = new_means

    return regions


def visualize_regions(image, regions, title="Regions"):
    """
    Visualize all regions with random colors.

    :param image: Grayscale input image.
    :param regions: List of regions, where each region is a set of (x, y) pixels.
    :param title: Title for the plot.
    """
    vis_image = np.zeros((*image.shape, 3), dtype=np.uint8)
    vis_image[:, :, 0] = image  # Add grayscale intensity to the red channel

    for region in regions:
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        for x, y in region:
            vis_image[y, x] = color

    plt.figure(figsize=(8, 8))
    plt.imshow(vis_image)
    plt.title(title)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    # Load test images
    inspected_image_path = "defective_examples/case1_inspected_image.tif"
    reference_image_path = "defective_examples/case1_reference_image.tif"

    inspected_image = cv2.imread(inspected_image_path, cv2.IMREAD_GRAYSCALE)
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)

    if inspected_image is None or reference_image is None:
        print("Error: Could not load the test images.")
        exit(1)

    for name, image in [("Inspected", inspected_image), ("Reference", reference_image)]:
        print(f"\nProcessing {name} Image...")

        # Step 1: Preprocess the image
        print("Preprocessing the image with two smoothing steps...")
        smoothed_image = preprocess_image(
            image, method1="gaussian", kernel_size1=5, method2="median", kernel_size2=3
        )
        plt.figure(figsize=(8, 8))
        plt.imshow(smoothed_image, cmap="gray")
        plt.title(f"{name} Image: Smoothed")
        plt.axis("off")
        plt.show()

        # Step 2: Perform full region growing
        print("Performing region growing...")
        regions = region_growing_with_full_coverage(smoothed_image, n_samples=10000, n_pixels=50, threshold=22.0)
        print(f"Initial number of regions: {len(regions)}")
        visualize_regions(smoothed_image, regions, title=f"{name} Image: Initial Regions")

        # Step 3: Merge similar regions
        print("Merging similar regions...")
        merged_regions = merge_similar_regions(smoothed_image, regions, similarity_threshold=10.0)
        print(f"Number of merged regions: {len(merged_regions)}")
        visualize_regions(smoothed_image, merged_regions, title=f"{name} Image: Merged Regions")

        # Step 4: Validate coverage (no pixel left behind)
        covered_pixels = set.union(*merged_regions) if merged_regions else set()
        assert len(covered_pixels) == smoothed_image.size, f"Not all pixels are covered in {name} image!"
        print(f"All pixels are covered in the {name} image.")
    # Load test images
    inspected_image_path = "defective_examples/case1_inspected_image.tif"
    reference_image_path = "defective_examples/case1_reference_image.tif"

    inspected_image = cv2.imread(inspected_image_path, cv2.IMREAD_GRAYSCALE)
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)

    if inspected_image is None or reference_image is None:
        print("Error: Could not load the test images.")
        exit(1)

    for name, image in [("Inspected", inspected_image), ("Reference", reference_image)]:
        print(f"\nProcessing {name} Image...")

        # Step 1: Preprocess the image
        print("Preprocessing the image...")
        smoothed_image = preprocess_image(image, method="gaussian", kernel_size=5)
        plt.figure(figsize=(8, 8))
        plt.imshow(smoothed_image, cmap="gray")
        plt.title(f"{name} Image: Smoothed")
        plt.axis("off")
        plt.show()

        # Step 2: Perform full region growing
        print("Performing region growing...")
        regions = region_growing_with_full_coverage(smoothed_image, n_samples=100, n_pixels=50, threshold=15.0)
        print(f"Initial number of regions: {len(regions)}")
        visualize_regions(smoothed_image, regions, title=f"{name} Image: Initial Regions")

        # Step 3: Merge similar regions
        print("Merging similar regions...")
        merged_regions = merge_similar_regions(smoothed_image, regions, similarity_threshold=10.0)
        print(f"Number of merged regions: {len(merged_regions)}")
        visualize_regions(smoothed_image, merged_regions, title=f"{name} Image: Merged Regions")

        # Step 4: Validate coverage (no pixel left behind)
        covered_pixels = set.union(*merged_regions) if merged_regions else set()
        assert len(covered_pixels) == smoothed_image.size, f"Not all pixels are covered in {name} image!"
        print(f"All pixels are covered in the {name} image.")