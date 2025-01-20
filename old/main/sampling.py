import random
import numpy as np

def random_pixel_sampling_no_overlap(
    image,
    visited,
    n_samples=500,
    n_pixels=20,
    top_percentage=5
):
    """
    Identify rectangles (area ~ n_pixels) in uncovered parts of 'image'.
    Sort them by variance (lower = more uniform), and pick the top few
    without overlapping each other.

    :param image: Grayscale image (H x W).
    :param visited: Boolean array (H x W), True where pixels are already covered.
    :param n_samples: Number of random rectangles to propose.
    :param n_pixels: Approx area of each rectangle.
    :param top_percentage: Keep only the top X% rectangles with lowest variance.
    :return: A list of rectangles, each a list of (x, y) pixel coords.
    """
    h, w = image.shape
    all_rects = []
    variances = []

    for _ in range(n_samples):
        # Pick a random top-left in an uncovered area (with some retries)
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