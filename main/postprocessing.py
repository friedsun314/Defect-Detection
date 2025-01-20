import numpy as np
import cv2
import matplotlib.pyplot as plt

def reconstruct_full_mask(partition_masks, positions, full_image_shape, overlap_strategy="max"):
    """
    Reconstructs the full binary mask from partitioned binary masks.

    Args:
        partition_masks (list of numpy.ndarray): List of binary masks for each partition.
        positions (list of tuple): List of (x, y, width, height) tuples indicating each partition's position.
        full_image_shape (tuple): Shape of the full image (height, width).
        overlap_strategy (str): Strategy for handling overlapping regions ('max' or 'average').

    Returns:
        numpy.ndarray: Full reconstructed binary mask.
    """
    full_mask = np.zeros(full_image_shape, dtype=np.uint8)
    weight_mask = np.zeros(full_image_shape, dtype=np.float32)  # To handle overlaps

    for mask, (x, y, width, height) in zip(partition_masks, positions):
        # Add the binary mask to the corresponding region of the full mask
        region = full_mask[y:y+height, x:x+width]
        weight_region = weight_mask[y:y+height, x:x+width]

        if overlap_strategy == "max":
            # Logical OR for overlapping regions
            np.maximum(region, mask, out=region)
        elif overlap_strategy == "average":
            # Average the overlapping regions
            weight_region += 1
            region += mask

    if overlap_strategy == "average":
        # Normalize the mask by dividing by the weight mask to compute the average
        with np.errstate(divide='ignore', invalid='ignore'):
            full_mask = np.divide(full_mask, weight_mask, where=(weight_mask > 0))
            full_mask = (full_mask > 0.5).astype(np.uint8) * 255  # Binarize after averaging

    return full_mask

if __name__ == "__main__":
    # Example usage
    from partitioning import partition_image

    # Paths to test image
    image_path = "data/defective_examples/case1_inspected_image.tif"

    # Load image
    full_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Ensure image is loaded
    if full_image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Simulate partition masks (Example: Random binary masks for demonstration purposes)
    partitions = (3, 3)
    overlap = 0.5
    subimages, positions = partition_image(full_image, partitions, overlap)
    partition_masks = [np.random.randint(0, 2, (h, w), dtype=np.uint8) * 255 for _, _, w, h in positions]

    # Reconstruct the full mask
    full_mask = reconstruct_full_mask(partition_masks, positions, full_image.shape, overlap_strategy="max")

    # Visualize the full reconstructed mask
    plt.figure(figsize=(10, 10))
    plt.title("Reconstructed Full Mask")
    plt.imshow(full_mask, cmap="gray")
    plt.axis("off")
    plt.show()