import numpy as np
from scipy.ndimage import shift
import matplotlib.pyplot as plt

def create_sub_images(image, overlap=0.9):
    """
    Create two overlapping sub-images from the input image with a meaningful size difference.

    Parameters:
        image (np.ndarray): Input image (H, W) or (H, W, C).
        overlap (float): Fraction of shared pixels between the two sub-images (0 to 1).

    Returns:
        tuple: (reference_image, inspected_image)
    """
    H, W = image.shape[:2]
    overlap_h = int(H * overlap)
    overlap_w = int(W * overlap)

    # Reference sub-image (larger)
    ref_image = image[:overlap_h, :overlap_w]

    # Inspected sub-image (smaller)
    insp_image = image[overlap_h // 2:overlap_h + overlap_h // 2,
                        overlap_w // 2:overlap_w + overlap_w // 2]

    return ref_image, insp_image

# --- Testing and Visualization ---
if __name__ == "__main__":
    import cv2

    # Create a better test image (black background with distinct shapes)
    test_image = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(test_image, (50, 50), (150, 100), 255, -1)  # White rectangle
    cv2.circle(test_image, (100, 150), 30, 255, -1)  # White circle

    # Generate sub-images
    ref_image, insp_image = create_sub_images(test_image, overlap=0.9)

    # Visualize the original and sub-images
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(test_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Reference Image")
    plt.imshow(ref_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Inspected Image")
    plt.imshow(insp_image, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()    # Use the same checkerboard test image
    test_image = np.zeros((100, 100), dtype=np.uint8)
    test_image[::2, ::2] = 255
    test_image[1::2, 1::2] = 255

    # Generate sub-images
    ref_image, insp_image = create_sub_images(test_image)

    # Visualize the original and sub-images
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(test_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Reference Image")
    plt.imshow(ref_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Inspected Image")
    plt.imshow(insp_image, cmap='gray')
    plt.axis('off')

    plt.show()