import numpy as np

def partition_image(image, partitions=(3, 3), overlap=0.2):
    """
    Partitions an image into smaller overlapping subimages.

    Args:
        image (numpy.ndarray): Input image to be partitioned.
        partitions (tuple): Number of partitions along (rows, cols).
        overlap (float): Overlap fraction between partitions (0 to 1).

    Returns:
        list: List of subimages (with overlap).
        list: List of (x, y, width, height) tuples indicating subimage positions.
    """
    rows, cols = partitions
    h, w = image.shape
    sub_h = h // rows
    sub_w = w // cols
    overlap_h = int(sub_h * overlap)
    overlap_w = int(sub_w * overlap)

    subimages = []
    positions = []

    for i in range(rows):
        for j in range(cols):
            x_start = max(0, j * sub_w - overlap_w)
            x_end = min(w, (j + 1) * sub_w + overlap_w)
            y_start = max(0, i * sub_h - overlap_h)
            y_end = min(h, (i + 1) * sub_h + overlap_h)

            subimage = image[y_start:y_end, x_start:x_end]
            subimages.append(subimage)
            positions.append((x_start, y_start, x_end - x_start, y_end - y_start))

    return subimages, positions

if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt

    # Paths to test images
    image_path = "data/defective_examples/case1_reference_image.tif"

    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Ensure image is loaded
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Partition the image
    partitions = (3, 3)
    overlap = 0.2
    subimages, positions = partition_image(image, partitions, overlap)
