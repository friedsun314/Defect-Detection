import random
import numpy as np
import cv2
import matplotlib.pyplot as plt


def random_pixel_sampling_no_overlap(image, visited, n_samples=1000, n_pixels=20, top_percentage=5):
    """
    Identify rectangles (area ~ n_pixels) in uncovered parts of 'image'.
    Sort them by variance (lower = more uniform), and pick the top few
    without overlapping each other.

    :param image: Grayscale image (H x W).
    :param visited: Boolean array (H x W), True where pixels are already covered.
    :param n_samples: Number of random rectangles to propose.
    :param n_pixels: Approximate area of each rectangle.
    :param top_percentage: Keep only the top X% rectangles with lowest variance.
    :return: A list of rectangles, each a list of (x, y) pixel coordinates.
    """
    h, w = image.shape
    all_rects = []
    variances = []

    for _ in range(n_samples):
        # Pick a random top-left corner in an uncovered area
        for _try in range(50):
            x_start = random.randint(0, w - 1)
            y_start = random.randint(0, h - 1)
            if not visited[y_start, x_start]:
                break
        else:
            continue  # Couldn't find an uncovered start

        # Find valid rectangle dimensions
        for _retry in range(50):
            rect_h = random.randint(1, n_pixels)
            rect_w = n_pixels // rect_h
            if x_start + rect_w <= w and y_start + rect_h <= h:
                # Collect uncovered pixels in this rectangle
                pixels = []
                for x in range(x_start, x_start + rect_w):
                    for y in range(y_start, y_start + rect_h):
                        if not visited[y, x]:
                            pixels.append((x, y))

                # Skip if too many pixels are already visited
                if len(pixels) < int(0.8 * n_pixels):
                    continue

                # Compute variance
                intensities = [image[py, px] for (px, py) in pixels]
                var = np.var(intensities)

                all_rects.append(pixels)
                variances.append(var)
                break

    if not all_rects:
        return []

    # Sort by ascending variance
    sorted_indices = np.argsort(variances)
    top_count = max(1, len(sorted_indices) * top_percentage // 100)
    chosen_indices = sorted_indices[:top_count]

    # Pick top rectangles without internal overlap
    used = set()
    final_rects = []
    for idx in chosen_indices:
        rect = all_rects[idx]
        if any(px in used for px in rect):
            continue
        final_rects.append(rect)
        for px in rect:
            used.add(px)

    return final_rects


# Visualization Helper
def visualize_sampled_rectangles(image, rectangles, title="Sampled Rectangles"):
    """
    Visualize sampled rectangles on the image with different colors.

    :param image: Grayscale input image.
    :param rectangles: List of rectangles, where each rectangle is a list of (x, y) pixel coordinates.
    :param title: Title for the plot.
    """
    vis_image = np.zeros((*image.shape, 3), dtype=np.uint8)
    vis_image[:, :, 0] = image  # Add grayscale intensity to the red channel

    for rect in rectangles:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for x, y in rect:
            vis_image[y, x] = color

    plt.figure(figsize=(6, 6))
    plt.imshow(vis_image)
    plt.title(title)
    plt.axis("off")
    plt.show()


# Embedded Tests with Real Images
if __name__ == "__main__":
    # Load test images
    inspected_image_path = "defective_examples/case1_inspected_image.tif"
    reference_image_path = "defective_examples/case1_reference_image.tif"

    inspected_image = cv2.imread(inspected_image_path, cv2.IMREAD_GRAYSCALE)
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)

    if inspected_image is None or reference_image is None:
        print("Error: Could not load the test images. Check the file paths.")
        exit(1)

    # Test 1: Sampling on the inspected image
    visited = np.zeros_like(inspected_image, dtype=bool)
    rectangles = random_pixel_sampling_no_overlap(inspected_image, visited, n_samples=10000, n_pixels=50)
    print(f"Number of rectangles sampled: {len(rectangles)}")
    visualize_sampled_rectangles(inspected_image, rectangles, title="Sampling on Inspected Image")

    # Test 2: Sampling on the reference image
    visited = np.zeros_like(reference_image, dtype=bool)
    rectangles = random_pixel_sampling_no_overlap(reference_image, visited, n_samples=1000, n_pixels=50)
    print(f"Number of rectangles sampled: {len(rectangles)}")
    visualize_sampled_rectangles(reference_image, rectangles, title="Sampling on Reference Image")