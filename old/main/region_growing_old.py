import numpy as np
from collections import deque

def region_growing(image, seed_pixels, visited, threshold=10.0, connectivity=8):
    """
    BFS-based region-growing from a set of seed_pixels.
    Updates 'visited' for all grown pixels. Returns a set of (x,y).

    :param image: Grayscale image (H x W).
    :param seed_pixels: List of (x, y) coords to start from.
    :param visited: Boolean array, updated here for covered pixels.
    :param threshold: Allowed intensity difference from the running mean.
    :param connectivity: 4 or 8 for neighbor definition.
    :return: A set of all pixels (x, y) in the grown region.
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

    # Compute initial mean
    intensities = [image[y, x] for (x, y) in seed_pixels]
    if not intensities:
        return set()
    region_mean = float(np.mean(intensities))

    region = set(seed_pixels)
    for (sx, sy) in seed_pixels:
        visited[sy, sx] = True

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

                    # Update region mean (online)
                    old_count = len(region) - 1
                    region_mean = (region_mean * old_count + val) / len(region)

    return region