import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import felzenszwalb
from skimage.color import label2rgb
import cv2

def segment_frame(frame, scale=100, sigma=0.5, min_size=50, visualize=False):
    """
    Segment a frame using Felzenszwalb-Huttenlocher's algorithm.

    Args:
        frame (np.ndarray): Grayscale or RGB frame.
        scale (float): Balances color-space size and image size. Higher scale means larger clusters.
        sigma (float): Width of Gaussian smoothing kernel for preprocessing.
        min_size (int): Minimum component size. Enforced using postprocessing.
        visualize (bool): Whether to visualize the segmented frame.

    Returns:
        segments (np.ndarray): Segmented frame with labels.
    """
    # Input validation
    if not isinstance(frame, np.ndarray):
        raise ValueError("Input frame must be a numpy array.")
    if frame.ndim not in [2, 3]:
        raise ValueError("Input frame must be 2D (grayscale) or 3D (RGB).")
    if frame.ndim == 3 and frame.shape[2] not in [3, 4]:  # RGB or RGBA
        raise ValueError("Input frame must have 3 or 4 channels if it's 3D.")

    # Perform segmentation
    segments = felzenszwalb(frame, scale=scale, sigma=sigma, min_size=min_size)
    
    # Validate output
    if not isinstance(segments, np.ndarray):
        raise RuntimeError("Segmentation failed to produce a valid output.")

    # Visualization
    if visualize:
        segmented_image = label2rgb(segments, frame, kind='avg')
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Frame")
        if frame.ndim == 3:
            plt.imshow(frame)
        else:
            plt.imshow(frame, cmap="gray")
        plt.axis("off")
        
        plt.subplot(1, 2, 2)
        plt.title("Segmented Frame")
        plt.imshow(segmented_image)
        plt.axis("off")
        
        plt.show()
    
    return segments

#############################
# Testing Section
#############################

def test_segment_frame_with_actual_image():
    """
    Test the segment_frame function with an actual inspected image.
    """
    # Hardcoded path to the inspected image
    inspected_image_path = "defective_examples/case1_inspected_image.tif"
    
    # Load the image
    image_insp = cv2.imread(inspected_image_path, cv2.IMREAD_GRAYSCALE)
    if image_insp is None:
        raise FileNotFoundError(f"Could not load image from {inspected_image_path}. Please check the path.")
    
    # Convert grayscale to RGB for segmentation visualization
    image_insp_rgb = cv2.cvtColor(image_insp, cv2.COLOR_GRAY2RGB)
    
    # Segment the image
    print("Segmenting the actual inspected image...")
    segment_frame(image_insp_rgb, scale=100, sigma=0.5, min_size=50, visualize=True)
    print("Segmentation completed for the actual inspected image.")

def test_segment_frame_with_visualization():
    """
    Test the segment_frame function with visualization.
    """
    # Test Case 1: Simple Grayscale Image
    print("Running Test Case 1: Simple Grayscale Image")
    frame_gray = np.array([
        [10, 10, 10, 10],
        [10, 20, 20, 10],
        [10, 20, 20, 10],
        [10, 10, 10, 10]
    ], dtype=np.uint8)
    segment_frame(frame_gray, visualize=True)

    # Test Case 2: Simple RGB Image
    print("Running Test Case 2: Simple RGB Image")
    frame_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_rgb[25:75, 25:75] = [255, 0, 0]  # Add a red square
    segment_frame(frame_rgb, visualize=True)

    # Test Case 3: Large Random Image
    print("Running Test Case 3: Large Random Image")
    frame_large = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    segment_frame(frame_large, visualize=True)

    print("All visualization tests passed.")

if __name__ == "__main__":
    print("Starting tests with visualization...")
    test_segment_frame_with_visualization()
    print("\nTesting with the actual inspected image...")
    test_segment_frame_with_actual_image()
    print("All tests completed successfully!")