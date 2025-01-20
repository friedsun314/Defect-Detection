import cv2
import numpy as np

# Load images
reference_image_path = "data/defective_examples/case1_reference_image.tif"
inspected_image_path = "data/defective_examples/case1_inspected_image.tif"

reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
inspected_image = cv2.imread(inspected_image_path, cv2.IMREAD_GRAYSCALE)

# Ensure images are loaded
if reference_image is None or inspected_image is None:
    raise FileNotFoundError("One or both image paths are incorrect.")

# Detect keypoints and descriptors
sift = cv2.SIFT_create()
keypoints_ref, descriptors_ref = sift.detectAndCompute(reference_image, None)
keypoints_ins, descriptors_ins = sift.detectAndCompute(inspected_image, None)

# Match descriptors using FLANN-based matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors_ref, descriptors_ins, k=2)

# Apply Lowe's ratio test
good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

# Compute homography if enough matches are found
MIN_MATCH_COUNT = 10
if len(good_matches) >= MIN_MATCH_COUNT:
    src_pts = np.float32([keypoints_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_ins[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
else:
    raise ValueError("Not enough matches found to compute the homography matrix.")

# Frame parameters
frame_size = 50  # Size of the frame in pixels
frame_pos = [100, 100]  # Initial position of the frame (x, y)

def update_reference_window(frame_pos):
    """Updates the reference image window based on the current frame position."""
    x, y = frame_pos
    points = np.array([
        [x, y],
        [x + frame_size, y],
        [x + frame_size, y + frame_size],
        [x, y + frame_size]
    ], dtype=np.float32).reshape(-1, 1, 2)

    # Map the frame's corners using the homography matrix
    transformed_points = cv2.perspectiveTransform(points, H)

    # Draw the corresponding square on the reference image
    temp_reference = reference_image.copy()
    cv2.polylines(temp_reference, [np.int32(transformed_points)], isClosed=True, color=255, thickness=2)
    cv2.imshow("Reference Image", temp_reference)

# Mouse callback function to move the frame
def move_frame(event, x, y, flags, param):
    global frame_pos
    if event == cv2.EVENT_MOUSEMOVE:
        frame_pos = [x - frame_size // 2, y - frame_size // 2]
        frame_pos[0] = max(0, min(frame_pos[0], inspected_image.shape[1] - frame_size))
        frame_pos[1] = max(0, min(frame_pos[1], inspected_image.shape[0] - frame_size))

        # Update the inspected image with the frame
        temp_inspected = inspected_image.copy()
        cv2.rectangle(temp_inspected, (frame_pos[0], frame_pos[1]),
                      (frame_pos[0] + frame_size, frame_pos[1] + frame_size), (255, 255, 255), 2)
        cv2.imshow("Inspected Image", temp_inspected)

        # Update the reference image window
        update_reference_window(frame_pos)

# Display the images
cv2.namedWindow("Inspected Image")
cv2.setMouseCallback("Inspected Image", move_frame)

# Initial display
move_frame(cv2.EVENT_MOUSEMOVE, frame_pos[0] + frame_size // 2, frame_pos[1] + frame_size // 2, None, None)

print("Move your mouse over the 'Inspected Image' window to control the frame.")
cv2.imshow("Reference Image", reference_image)
cv2.waitKey(0)
cv2.destroyAllWindows() 