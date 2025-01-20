import cv2
import numpy as np
import unittest

def match_keypoints(image_ref, image_insp, min_match_count=10):
    """
    Find keypoints and compute the homography between two images.
    
    Parameters:
    - image_ref: The reference image (grayscale).
    - image_insp: The inspected image (grayscale).
    - min_match_count: Minimum number of matches required to compute a homography.
    
    Returns:
    - H: The homography matrix (or None if not enough matches or no keypoints).
    - kp1: Keypoints in the reference image.
    - kp2: Keypoints in the inspected image.
    - matches: List of matches between the images.
    """
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image_ref, None)
    kp2, des2 = orb.detectAndCompute(image_insp, None)

    if des1 is None or des2 is None:
        return None, [], [], []

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < min_match_count:
        return None, kp1, kp2, matches

    # Extract points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H, kp1, kp2, matches


# Test Suite
class TestMatching(unittest.TestCase):
    def setUp(self):
        # Create two simple test images with a matching feature
        self.image_ref = np.zeros((100, 100), dtype=np.uint8)
        self.image_insp = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(self.image_ref, (50, 50), 10, 255, -1)
        cv2.circle(self.image_insp, (50, 50), 10, 255, -1)

    def test_match_keypoints_success(self):
        """Test if keypoints are matched successfully with simple images."""
        H, kp1, kp2, matches = match_keypoints(self.image_ref, self.image_insp)
        self.assertIsNotNone(H, "Homography matrix should not be None.")
        self.assertGreater(len(matches), 0, "There should be at least one match.")

    def test_match_keypoints_failure(self):
        """Test if function handles cases with no matches."""
        # Modify the inspected image to have no matching features
        self.image_insp = np.zeros((100, 100), dtype=np.uint8)
        H, kp1, kp2, matches = match_keypoints(self.image_ref, self.image_insp, min_match_count=1)
        self.assertIsNone(H, "Homography matrix should be None for no matches.")
        self.assertEqual(len(matches), 0, "There should be no matches.")

    def test_min_match_count(self):
        """Test if min_match_count parameter is respected."""
        H, kp1, kp2, matches = match_keypoints(self.image_ref, self.image_insp, min_match_count=50)
        self.assertIsNone(H, "Homography matrix should be None for insufficient matches.")

# Run tests if this script is executed directly
if __name__ == "__main__":
    unittest.main()