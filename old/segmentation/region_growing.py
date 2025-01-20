import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import cv2
from seed_sampling import random_pixel_sampling_no_overlap


def region_growing_with_full_coverage(image, n_samples=500, n_pixels=20, threshold=10.0, connectivity=8):
    """
    Perform region growing using seeds sampled from the image and ensure all pixels are covered.

    :param image: Grayscale image (H x W).
    :param n_samples: Number of random seeds to sample.
    :param n_pixels: Approx area of each seed rectangle.
    :param threshold: Allowed intensity difference from the running mean.
    :param connectivity: 4 or 8 for neighbor definition.
    :return: A list of grown regions, each a set of (x, y) pixel coordinates.
    """
    visited = np.zeros_like(image, dtype=bool)
    all_regions = []

    # Step 1: Initial region growing
    regions = region_growing(image, visited, n_samples=n_samples, n_pixels=n_pixels, threshold=threshold, connectivity=connectivity)
    all_regions.extend(regions)

    # Step 2: Check for uncovered pixels
    all_pixels = {(x, y) for y in range(image.shape[0]) for x in range(image.shape[1])}
    covered_pixels = set.union(*all_regions) if all_regions else set()
    uncovered_pixels = all_pixels - covered_pixels

    # Step 3: Handle uncovered pixels
    for pixel in uncovered_pixels:
        if not visited[pixel[1], pixel[0]]:
            new_region = grow_region_from_seed(image, visited, pixel, threshold, connectivity)
            all_regions.append(new_region)

    return all_regions


def region_growing(image, visited, n_samples=500, n_pixels=20, threshold=10.0, connectivity=8):
    """
    Perform region growing using seeds sampled from the image.

    :param image: Grayscale image (H x W).
    :param visited: Boolean array, updated here for covered pixels.
    :param n_samples: Number of random seeds to sample.
    :param n_pixels: Approx area of each seed rectangle.
    :param threshold: Allowed intensity difference from the running mean.
    :param connectivity: 4 or 8 for neighbor definition.
    :return: A list of grown regions, each a set of (x, y) pixel coordinates.
    """
    h, w = image.shape

    # Neighborhood offsets
    if connectivity == 4:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]

    def in_bounds(x, y):
        return 0 <= x < w and 0 <= y < h

    all_regions = []

    # Step 1: Sample seeds
    sampled_rectangles = random_pixel_sampling_no_overlap(
        image, visited, n_samples=n_samples, n_pixels=n_pixels
    )

    # Step 2: Grow regions from seeds
    for rect in sampled_rectangles:
        seed_pixels = [(x, y) for (x, y) in rect if not visited[y, x]]
        if not seed_pixels:
            continue

        # Initialize region
        intensities = [image[y, x] for (x, y) in seed_pixels]
        region_mean = float(np.mean(intensities))
        region = set(seed_pixels)

        # Mark seeds as visited
        for (sx, sy) in seed_pixels:
            visited[sy, sx] = True

        # BFS for region growing
        queue = deque(seed_pixels)
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if in_bounds(nx, ny) and not visited[ny, nx]:
                    val = image[ny, nx]
                    if abs(val - region_mean) <= threshold:
                        visited[ny, nx] = True
                        region.add((nx, ny))
                        queue.append((nx, ny))

                        # Incrementally update region mean
                        old_count = len(region) - 1
                        region_mean = (region_mean * old_count + val) / len(region)

        if region:
            all_regions.append(region)

    return all_regions


def grow_region_from_seed(image, visited, seed_pixel, threshold, connectivity):
    """
    Grow a single region from a given seed pixel.

    :param image: Grayscale image (H x W).
    :param visited: Boolean array to track visited pixels.
    :param seed_pixel: Starting pixel (x, y) for region growing.
    :param threshold: Allowed intensity difference from the running mean.
    :param connectivity: 4 or 8 for neighbor definition.
    :return: A set of (x, y) pixels in the grown region.
    """
    h, w = image.shape

    # Neighborhood offsets
    if connectivity == 4:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]

    def in_bounds(x, y):
        return 0 <= x < w and 0 <= y < h

    # Initialize region
    region = {seed_pixel}
    intensities = [image[seed_pixel[1], seed_pixel[0]]]
    region_mean = float(np.mean(intensities))
    queue = deque([seed_pixel])
    visited[seed_pixel[1], seed_pixel[0]] = True

    # Grow region
    while queue:
        cx, cy = queue.popleft()
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if in_bounds(nx, ny) and not visited[ny, nx]:
                val = image[ny, nx]
                if abs(val - region_mean) <= threshold:
                    visited[ny, nx] = True
                    region.add((nx, ny))
                    queue.append((nx, ny))

                    # Incrementally update mean
                    old_count = len(region) - 1
                    region_mean = (region_mean * old_count + val) / len(region)

    return region


def visualize_regions(image, regions, title="Region Growing"):
    """
    Visualize all grown regions with random colors.

    :param image: Grayscale input image.
    :param regions: List of regions, where each region is a set of (x, y) pixels.
    :param title: Title for the plot.
    """
    vis_image = np.zeros((*image.shape, 3), dtype=np.uint8)
    vis_image[:, :, 0] = image  # Add grayscale to the red channel

    for region in regions:
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        for x, y in region:
            vis_image[y, x] = color

    plt.figure(figsize=(8, 8))
    plt.imshow(vis_image)
    plt.title(title)
    plt.axis("off")
    plt.show()


# Tests
if __name__ == "__main__":
    # Load test images
    inspected_image_path = "defective_examples/case1_inspected_image.tif"

    inspected_image = cv2.imread(inspected_image_path, cv2.IMREAD_GRAYSCALE)
    if inspected_image is None:
        print("Error: Could not load the inspected image.")
        exit(1)

    # Full coverage test
    regions = region_growing_with_full_coverage(inspected_image, n_samples=10000, n_pixels=50, threshold=25.0)
    covered_pixels = set.union(*regions) if regions else set()
    assert len(covered_pixels) == inspected_image.size, "Not all pixels are covered!"

    print(f"Total regions: {len(regions)}")
    visualize_regions(inspected_image, regions, title="Region Growing with Full Coverage")