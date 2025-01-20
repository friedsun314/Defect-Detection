# main.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

from sampling import random_pixel_sampling_no_overlap
from region_growing_old import region_growing
from merging import merge_similar_regions

def visualize_regions(image, regions, title="Regions"):
    """
    Overlays each region in a random color.
    """
    h, w = image.shape
    color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for reg in regions:
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
        for (x, y) in reg:
            color_img[y, x] = color

    plt.figure(figsize=(6,6))
    plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

def iterative_full_segmentation(
    image_path,
    max_iterations=50,
    threshold=3.0,
    similarity_threshold=3.0,
    n_samples=1000,
    n_pixels=20,
    top_percentage=1
):
    """
    1) Load grayscale image
    2) While uncovered pixels remain (or until max_iterations):
       - sample no-overlap seeds
       - region-grow each seed
    3) Merge similar regions
    4) Return final regions
    """
    # 1) Load
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    h, w = img_gray.shape
    visited = np.zeros((h, w), dtype=bool)

    all_regions = []

    for _ in range(max_iterations):
        if visited.all():
            break  # fully covered

        # 2A) Sample seeds
        rectangles = random_pixel_sampling_no_overlap(
            img_gray,
            visited=visited,
            n_samples=n_samples,
            n_pixels=n_pixels,
            top_percentage=top_percentage
        )

        if not rectangles:
            # No more new seeds => no progress => stop
            break

        # 2B) Region-grow
        for rect in rectangles:
            # Filter out already visited
            seed_pix = [(x, y) for (x, y) in rect if not visited[y, x]]
            if not seed_pix:
                continue
            region = region_growing(
                image=img_gray,
                seed_pixels=seed_pix,
                visited=visited,
                threshold=threshold,
                connectivity=8
            )
            if region:
                all_regions.append(region)

    # 3) Merge similar
    merged_regions = merge_similar_regions(img_gray, all_regions, similarity_threshold)

    return merged_regions, img_gray

if __name__ == "__main__":
    # Example usage
    image_path = "defective_examples/case1_inspected_image.tif"  # replace with an actual path
    final_regions, gray_image = iterative_full_segmentation(
        image_path=image_path,
        max_iterations=15,
        threshold=20.0,
        similarity_threshold=30.0,
        n_samples=1000,
        n_pixels=20,
        top_percentage=1
    )

    print(f"Found {len(final_regions)} regions after merging.")

    # Visualize
    visualize_regions(gray_image, final_regions, title="Final Merged Regions")