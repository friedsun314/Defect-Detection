import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import felzenszwalb, slic
from skimage.color import label2rgb


def segment_with_felzenszwalb(image, scale=100, sigma=0.5, min_size=50):
    """Segment the image using Felzenszwalb's algorithm."""
    segments = felzenszwalb(image, scale=scale, sigma=sigma, min_size=min_size)
    return segments


def segment_with_slic(image, n_segments=250, compactness=10, sigma=1):
    """Segment the image using SLIC (Simple Linear Iterative Clustering)."""
    # For grayscale images, set channel_axis to None
    segments = slic(image, n_segments=n_segments, compactness=compactness, sigma=sigma, start_label=1, channel_axis=None)
    return segments


def segment_frame(image, method="felzenszwalb", **kwargs):
    """
    Wrapper for segmentation methods.

    Args:
        image (np.ndarray): Input image (grayscale or RGB).
        method (str): Segmentation method. Options are "felzenszwalb" or "slic".
        **kwargs: Additional parameters for the segmentation method.

    Returns:
        np.ndarray: Segmentation labels.
    """
    if method == "felzenszwalb":
        return segment_with_felzenszwalb(image, **kwargs)
    elif method == "slic":
        return segment_with_slic(image, **kwargs)
    else:
        raise ValueError(f"Unknown segmentation method: {method}")


def visualize_segmentation(image, segments, title):
    """Visualize the segmentation result."""
    segmented_image = label2rgb(segments, image, kind="avg", bg_label=0)
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.imshow(segmented_image, cmap="gray")
    plt.axis("off")
    plt.show()


def test_segmentation_methods(image_path):
    """Test various segmentation methods on the given image."""
    # Load the inspected image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    print("Visualizing Felzenszwalb segmentation...")
    segments_felzenszwalb = segment_with_felzenszwalb(image, scale=100, sigma=0.5, min_size=50)
    visualize_segmentation(image, segments_felzenszwalb, "Felzenszwalb Segmentation")

    print("Visualizing SLIC segmentation...")
    segments_slic = segment_with_slic(image, n_segments=250, compactness=10, sigma=1)
    visualize_segmentation(image, segments_slic, "SLIC Segmentation")


if __name__ == "__main__":
    # Hardcoded path to the inspected image
    inspected_image_path = "defective_examples/case1_inspected_image.tif"
    test_segmentation_methods(inspected_image_path)