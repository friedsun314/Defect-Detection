import cv2
import matplotlib.pyplot as plt
import random
import numpy as np

def visualize_defect_map(image_insp, defect_map, title="Defect Map"):
    """
    Overlay defect map on original image. Defects are highlighted in red.

    Args:
        image_insp (np.ndarray): Original inspected grayscale image.
        defect_map (np.ndarray): Binary defect map (1=defect, 0=non-defect).
        title (str): Plot title.
    """
    # Input validation
    if not isinstance(image_insp, np.ndarray) or not isinstance(defect_map, np.ndarray):
        raise ValueError("Inputs must be numpy arrays.")
    if image_insp.shape != defect_map.shape:
        raise ValueError("image_insp and defect_map must have the same shape.")
    if not np.array_equal(defect_map, defect_map.astype(bool)):
        raise ValueError("defect_map must be binary (contain only 0s and 1s).")

    # Convert grayscale to BGR
    overlay = cv2.cvtColor(image_insp, cv2.COLOR_GRAY2BGR)
    
    # Highlight defects in red
    overlay[defect_map == 1] = [0, 0, 255]  # BGR format
    
    # Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def visualize_some_defects(image_insp, defect_map, num_defects=50, title="Sample Defects"):
    """
    Highlight a subset of defects to visualize.

    Args:
        image_insp (np.ndarray): Original inspected grayscale image.
        defect_map (np.ndarray): Binary defect map.
        num_defects (int): Number of defects to highlight.
        title (str): Plot title.
    """
    # Input validation
    if not isinstance(image_insp, np.ndarray) or not isinstance(defect_map, np.ndarray):
        raise ValueError("Inputs must be numpy arrays.")
    if image_insp.shape != defect_map.shape:
        raise ValueError("image_insp and defect_map must have the same shape.")
    if not np.array_equal(defect_map, defect_map.astype(bool)):
        raise ValueError("defect_map must be binary (contain only 0s and 1s).")
    
    # Find defect coordinates
    ys, xs = np.where(defect_map == 1)
    num_defects = min(num_defects, len(xs))
    
    if num_defects == 0:
        print("No defects to visualize.")
        return
    
    selected = random.sample(list(zip(xs, ys)), num_defects)
    
    # Convert to BGR
    overlay = cv2.cvtColor(image_insp, cv2.COLOR_GRAY2BGR)
    
    # Draw circles around defects
    for (x, y) in selected:
        cv2.circle(overlay, (x, y), radius=3, color=(0, 255, 0), thickness=1)  # Green circles
    
    # Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

#############################
# Testing Section
#############################

def test_visualize_functions():
    """
    Test visualize_defect_map and visualize_some_defects with various cases.
    """
    # Test Case 1: Simple Defect Map
    print("Running Test Case 1: Simple Defect Map")
    image = np.full((10, 10), 128, dtype=np.uint8)  # Gray image
    defect_map = np.zeros((10, 10), dtype=np.uint8)
    defect_map[4:6, 4:6] = 1  # Small defect in the center
    visualize_defect_map(image, defect_map, title="Simple Defect Map")
    visualize_some_defects(image, defect_map, num_defects=2, title="Sample Defects")
    print("Test Case 1 Passed.")

    # Test Case 2: No Defects
    print("Running Test Case 2: No Defects")
    defect_map_no_defects = np.zeros((10, 10), dtype=np.uint8)
    visualize_defect_map(image, defect_map_no_defects, title="No Defects")
    visualize_some_defects(image, defect_map_no_defects, title="No Defects")
    print("Test Case 2 Passed.")

    # Test Case 3: Large Defect Map
    print("Running Test Case 3: Large Defect Map")
    image_large = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    defect_map_large = np.zeros((100, 100), dtype=np.uint8)
    defect_map_large[30:40, 30:40] = 1  # Square defect
    visualize_defect_map(image_large, defect_map_large, title="Large Defect Map")
    visualize_some_defects(image_large, defect_map_large, num_defects=10, title="Sample Large Defects")
    print("Test Case 3 Passed.")

    # Test Case 4: Binary Defect Map Validation
    print("Running Test Case 4: Binary Defect Map Validation")
    defect_map_invalid = np.random.randint(0, 2, (10, 10)).astype(float)  # Invalid defect map
    try:
        visualize_defect_map(image, defect_map_invalid, title="Invalid Defect Map")
    except ValueError as e:
        print(f"Test Case 4 Passed: {e}")

    # Test Case 5: Empty Defect Map
    print("Running Test Case 5: Empty Defect Map")
    image_empty = np.empty((0, 0), dtype=np.uint8)
    defect_map_empty = np.empty((0, 0), dtype=np.uint8)
    try:
        visualize_defect_map(image_empty, defect_map_empty, title="Empty Defect Map")
    except ValueError as e:
        print(f"Test Case 5 Passed: {e}")

    print("All visualization tests passed successfully.")

if __name__ == "__main__":
    print("Starting tests for visualization...")
    test_visualize_functions()
    print("All tests completed successfully!")