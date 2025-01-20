import numpy as np
from skimage.transform import pyramid_gaussian
from region_growing import region_growing_with_full_coverage
from region_merging import merge_similar_regions
import matplotlib.pyplot as plt
import cv2


def construct_pyramid(image, max_layers=3, downscale=2):
    """
    Construct a Gaussian pyramid for an image.

    :param image: Grayscale input image.
    :param max_layers: Number of pyramid layers.
    :param downscale: Downscale factor for the pyramid.
    :return: List of images in the pyramid.
    """
    return list(pyramid_gaussian(image, max_layer=max_layers, downscale=downscale))


def segment_pyramid(image, max_layers=3, downscale=2, threshold=15.0, merge_threshold=10.0):
    """
    Perform segmentation using a Gaussian pyramid.

    :param image: Grayscale input image.
    :param max_layers: Number of pyramid layers.
    :param downscale: Downscale factor for the pyramid.
    :param threshold: Intensity threshold for region growing.
    :param merge_threshold: Threshold for merging similar regions.
    :return: Segmentation results across pyramid levels.
    """
    # Step 1: Construct the pyramid
    pyramid = construct_pyramid(image, max_layers=max_layers, downscale=downscale)
    pyramid_regions = []

    # Step 2: Segment at each pyramid level
    for level, layer in enumerate(pyramid):
        print(f"Processing Pyramid Level {level}, Shape: {layer.shape}")
        visited = np.zeros(layer.shape, dtype=bool)
        regions = region_growing_with_full_coverage(
        layer, visited, n_samples=100, n_pixels=50, threshold=threshold
        )
        merged_regions = merge_similar_regions(layer, regions, similarity_threshold=merge_threshold)
        pyramid_regions.append(merged_regions)

    return pyramid_regions, pyramid


def visualize_pyramid_segmentation(pyramid, pyramid_regions):
    """
    Visualize the segmentation at each pyramid level.

    :param pyramid: List of pyramid images.
    :param pyramid_regions: List of segmented regions for each pyramid level.
    """
    for level, (image, regions) in enumerate(zip(pyramid, pyramid_regions)):
        vis_image = np.zeros((*image.shape, 3), dtype=np.uint8)

        for region in regions:
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            for x, y in region:
                vis_image[y, x] = color

        plt.figure(figsize=(8, 8))
        plt.imshow(vis_image)
        plt.title(f"Pyramid Level {level} Segmentation")
        plt.axis("off")
        plt.show()


# Embedded Tests
if __name__ == "__main__":
    # Load reference and inspected images
    inspected_image_path = "defective_examples/case1_inspected_image.tif"
    reference_image_path = "defective_examples/case1_reference_image.tif"

    inspected_image = cv2.imread(inspected_image_path, cv2.IMREAD_GRAYSCALE)
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)

    if inspected_image is None or reference_image is None:
        print("Error: Could not load one or both images.")
        exit(1)

    # Parameters
    max_layers = 3
    threshold = 15.0
    merge_threshold = 10.0

    # Segmentation for inspected image
    print("Processing Inspected Image...")
    inspected_regions, inspected_pyramid = segment_pyramid(
        inspected_image, max_layers=max_layers, threshold=threshold, merge_threshold=merge_threshold
    )
    visualize_pyramid_segmentation(inspected_pyramid, inspected_regions)

    # Segmentation for reference image
    print("Processing Reference Image...")
    reference_regions, reference_pyramid = segment_pyramid(
        reference_image, max_layers=max_layers, threshold=threshold, merge_threshold=merge_threshold
    )
    visualize_pyramid_segmentation(reference_pyramid, reference_regions)