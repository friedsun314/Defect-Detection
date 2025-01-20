import numpy as np
import cv2

def sliding_window_mse(reference_region, inspected_frame):
    h_ref, w_ref = reference_region.shape
    h_frame, w_frame = inspected_frame.shape

    if h_frame > h_ref or w_frame > w_ref:
        raise ValueError("Inspected frame size must be smaller than or equal to the reference region size.")

    min_mse = float('inf')
    min_coords = None

    for y in range(h_ref - h_frame + 1):
        for x in range(w_ref - w_frame + 1):
            # Extract the current region from the reference region
            current_region = reference_region[y:y + h_frame, x:x + w_frame]

            # Compute the MSE between the inspected frame and the current region
            mse = np.mean((inspected_frame - current_region) ** 2)

            print(f"Sliding Window ({x}, {y})")
            print(f"Current Region:\n{current_region}")
            print(f"MSE: {mse}")

            # Update the minimum MSE and coordinates if a lower MSE is found
            if mse < min_mse:
                min_mse = mse
                min_coords = (x, y)

    print(f"Lowest MSE: {min_mse} at {min_coords}")
    return min_mse, min_coords



# Test cases
def test_sliding_window_mse():
    # Test case 1: Perfect match
    reference_region = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=np.uint8)
    inspected_frame = np.array([
        [5, 6],
        [8, 9]
    ], dtype=np.uint8)
    min_mse, min_coords = sliding_window_mse(reference_region, inspected_frame)
    assert min_coords == (1, 1), f"Expected (1, 1), got {min_coords}"
    assert min_mse == 0, f"Expected 0, got {min_mse}"

    # Test case 2: Different values
    reference_region = np.array([
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90]
    ], dtype=np.uint8)
    inspected_frame = np.array([
        [25, 35],
        [55, 65]
    ], dtype=np.uint8)
    min_mse, min_coords = sliding_window_mse(reference_region, inspected_frame)

    # Expected coordinates and MSE
    expected_coords = (1, 0)
    expected_mse = np.mean((inspected_frame - np.array([[20, 30], [50, 60]])) ** 2)

    # Debugging prints
    print(f"Inspected Frame:\n{inspected_frame}")
    print(f"Best Matching Region:\n{np.array([[40, 50], [70, 80]])}")
    print(f"Computed MSE: {expected_mse}")
    print(f"Actual MSE: {min_mse}")
    print(f"Min Coords: {min_coords}")

    assert min_coords == expected_coords, f"Expected {expected_coords}, got {min_coords}"
    assert np.isclose(min_mse, expected_mse, atol=1e-3), f"Expected MSE {expected_mse}, got {min_mse}"
    # Test case 3: Inspected frame larger than reference region
    reference_region = np.array([
        [1, 2],
        [3, 4]
    ], dtype=np.uint8)
    inspected_frame = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=np.uint8)
    try:
        min_mse, min_coords = sliding_window_mse(reference_region, inspected_frame)
    except ValueError as e:
        assert str(e) == "Inspected frame size must be smaller than or equal to the reference region size."

    # Test case 4: Identical frames
    reference_region = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=np.uint8)
    inspected_frame = reference_region
    min_mse, min_coords = sliding_window_mse(reference_region, inspected_frame)
    assert min_coords == (0, 0), f"Expected (0, 0), got {min_coords}"
    assert min_mse == 0, f"Expected 0, got {min_mse}"

    print("All tests passed!")


if __name__ == "__main__":
    # Run tests
    test_sliding_window_mse()

    # Example usage
    reference_region = np.array([
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90]
    ], dtype=np.uint8)
    inspected_frame = np.array([
        [25, 35],
        [55, 65]
    ], dtype=np.uint8)

    min_mse, min_coords = sliding_window_mse(reference_region, inspected_frame)
    print(f"Lowest MSE: {min_mse}")
    print(f"Coordinates with lowest MSE: {min_coords}")