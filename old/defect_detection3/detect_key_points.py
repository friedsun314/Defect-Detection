import cv2
import matplotlib.pyplot as plt

def detect_key_points(image, method="SIFT", min_blob_area=50, max_blob_area=500):
    """
    Detect key points by combining SIFT/ORB and blob detection.
    
    Args:
        image: The input grayscale image.
        method: The feature detection method ("SIFT" or "ORB").
        min_blob_area: Minimum area for blob detection.
        max_blob_area: Maximum area for blob detection.
        
    Returns:
        keypoints_combined: A list of combined key points.
        descriptors_detected: Descriptors from SIFT/ORB (for feature matching).
    """
    # Detect key points using SIFT/ORB
    if method == "SIFT":
        detector = cv2.SIFT_create()
    elif method == "ORB":
        detector = cv2.ORB_create()
    else:
        raise ValueError("Unsupported key point detection method.")
    
    keypoints_detected, descriptors_detected = detector.detectAndCompute(image, None)

    # Detect blobs in the image
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = min_blob_area
    params.maxArea = max_blob_area
    params.filterByCircularity = False  # Include non-circular blobs
    params.filterByConvexity = False
    params.filterByInertia = False

    blob_detector = cv2.SimpleBlobDetector_create(params)
    keypoints_blob = blob_detector.detect(image)

    # Combine the two sets of key points
    keypoints_combined = keypoints_detected + keypoints_blob

    return keypoints_combined, descriptors_detected

# Load the inspected image
inspected_image_path = "data/defective_examples/case1_inspected_image.tif"
inspected_image = cv2.imread(inspected_image_path, cv2.IMREAD_GRAYSCALE)

if inspected_image is None:
    raise FileNotFoundError(f"Failed to load image from {inspected_image_path}")

# Detect key points
method = "SIFT"  # You can also try "ORB"
keypoints_ins, descriptors_ins = detect_key_points(inspected_image, method=method)

# Visualize the key points
def visualize_keypoints(image, keypoints, title="Key Points"):
    """Visualize key points on the image."""
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure(figsize=(8, 8))
    plt.title(title)
    plt.imshow(image_with_keypoints, cmap='gray')
    plt.axis("off")
    plt.show()

# Visualize the detected key points
visualize_keypoints(inspected_image, keypoints_ins, title=f"Detected Key Points using {method}")