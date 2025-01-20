import numpy as np
import unittest

def center_reference(reference, inspected_shape):
    """
    Center the reference image in a larger matrix of zeros matching the inspected_shape.

    Parameters:
        reference (np.ndarray): The reference image (H_ref, W_ref).
        inspected_shape (tuple): Shape of the inspected image (H_insp, W_insp).

    Returns:
        np.ndarray: A larger matrix with the reference image centered.
    """
    H_ref, W_ref = reference.shape
    H_insp, W_insp = inspected_shape

    if H_insp < H_ref or W_insp < W_ref:
        raise ValueError("Inspected shape must be larger than or equal to reference shape.")

    # Initialize the centered matrix with zeros
    centered_matrix = np.zeros(inspected_shape, dtype=reference.dtype)

    # Calculate padding using floor division
    pad_y_start = (H_insp - H_ref) // 2
    pad_x_start = (W_insp - W_ref) // 2

    # Calculate end indices
    pad_y_end = pad_y_start + H_ref
    pad_x_end = pad_x_start + W_ref

    # Place the reference image at the calculated position
    centered_matrix[pad_y_start:pad_y_end, pad_x_start:pad_x_end] = reference

    # Verify that the first pixel of the reference image aligns correctly
    assert centered_matrix[pad_y_start, pad_x_start] == reference[0, 0], (
        f"Alignment check failed: centered_matrix[{pad_y_start}, {pad_x_start}] != reference[0, 0]"
    )

    return centered_matrix

class TestCenterReference(unittest.TestCase):
    def test_centering_basic(self):
        reference = np.array([[1, 2], [3, 4]])
        inspected_shape = (6, 6)
        expected = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 2, 0, 0],
            [0, 0, 3, 4, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ])
        result = center_reference(reference, inspected_shape)
        np.testing.assert_array_equal(result, expected)

    def test_centering_odd_padding(self):
        reference = np.array([[5]])
        inspected_shape = (3, 3)
        expected = np.array([
            [0, 0, 0],
            [0, 5, 0],
            [0, 0, 0],
        ])
        result = center_reference(reference, inspected_shape)
        np.testing.assert_array_equal(result, expected)

    def test_inspected_smaller_than_reference(self):
        reference = np.array([[1, 2, 3], [4, 5, 6]])
        inspected_shape = (1, 2)
        with self.assertRaises(ValueError):
            center_reference(reference, inspected_shape)

    def test_inspected_equal_to_reference(self):
        reference = np.array([[7, 8], [9, 10]])
        inspected_shape = (2, 2)
        expected = np.array([
            [7, 8],
            [9, 10],
        ])
        result = center_reference(reference, inspected_shape)
        np.testing.assert_array_equal(result, expected)

    def test_non_divisible_padding(self):
        reference = np.array([[1, 2], [3, 4]])
        inspected_shape = (5, 5)
        expected = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 2, 0, 0],
            [0, 3, 4, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])
        result = center_reference(reference, inspected_shape)
        np.testing.assert_array_equal(result, expected)

    def test_dtype_preservation(self):
        reference = np.array([[True, False], [False, True]], dtype=bool)
        inspected_shape = (4, 4)
        expected = np.array([
            [False, False, False, False],
            [False, True, False, False],
            [False, False, True, False],
            [False, False, False, False],
        ], dtype=bool)
        result = center_reference(reference, inspected_shape)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.dtype, reference.dtype)

    def test_non_divisible_padding_odd_dimensions(self):
        reference = np.array([[1, 2], [3, 4]])
        inspected_shape = (5, 7)
        expected = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 2, 0, 0, 0],
            [0, 0, 3, 4, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ])
        result = center_reference(reference, inspected_shape)
        np.testing.assert_array_equal(result, expected)

    def test_reference_larger_inspected_shape(self):
        reference = np.array([[1, 2, 3], [4, 5, 6]])
        inspected_shape = (6, 7)
        expected = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 2, 3, 0, 0],
            [0, 0, 4, 5, 6, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ])
        result = center_reference(reference, inspected_shape)
        np.testing.assert_array_equal(result, expected)

    def test_reference_single_row(self):
        reference = np.array([[1, 2, 3]])
        inspected_shape = (5, 5)
        expected = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 2, 3, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])
        result = center_reference(reference, inspected_shape)
        np.testing.assert_array_equal(result, expected)

    def test_reference_single_column(self):
        reference = np.array([[1], [2], [3]])
        inspected_shape = (7, 5)
        expected = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 2, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])
        result = center_reference(reference, inspected_shape)
        np.testing.assert_array_equal(result, expected)

    def test_reference_non_square_inspected_shape(self):
        reference = np.array([[1, 2], [3, 4]])
        inspected_shape = (4, 5)
        expected = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 2, 0, 0],
            [0, 3, 4, 0, 0],
            [0, 0, 0, 0, 0],
        ])
        result = center_reference(reference, inspected_shape)
        np.testing.assert_array_equal(result, expected)

    def test_large_reference_and_inspected_shape(self):
        reference = np.ones((10, 10), dtype=int)
        inspected_shape = (20, 20)
        expected = np.zeros((20, 20), dtype=int)
        expected[5:15, 5:15] = 1
        result = center_reference(reference, inspected_shape)
        np.testing.assert_array_equal(result, expected)

if __name__ == "__main__":
    unittest.main()