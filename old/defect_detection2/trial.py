import cv2
import numpy as np
import random
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

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
frame_size = 20  # Size of the frame in pixels
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

# Distance calculation function
def calculate_frame_distance(frame1, frame2):
    """Calculate three distance metrics between two frames and return their maximum."""
    # 1. Absolute Difference
    abs_diff = np.abs(frame1 - frame2).mean()
    
    # 2. Mean Squared Error (MSE)
    mse = np.mean((frame1 - frame2) ** 2)
    
    # 3. Structural Similarity Index (SSIM)
    ssim_value, _ = ssim(frame1, frame2, full=True)
    ssim_diff = 1 - ssim_value  # SSIM is similarity, we invert it to represent distance
    
    # Return the maximum distance
    return max(abs_diff, mse, ssim_diff)

# Threshold calculation function
def calculate_threshold(sampled_frames):
    """Calculate the threshold based on the maximum of multiple distance measures."""
    distances = []
    for inspected_frame, reference_frame in sampled_frames:
        distance = calculate_frame_distance(inspected_frame, reference_frame)
        distances.append(distance)
    threshold = np.percentile(distances, 10)  # Set threshold at 90th percentile
    return threshold

# Sampling frames
def sample_frames(inspected_image, reference_image, H, num_samples, frame_sizes):
    """Samples random frames from the inspected image and maps them to the reference image."""
    sampled_frames = []
    for _ in range(num_samples):
        frame_size = random.choice(frame_sizes)
        x = random.randint(0, inspected_image.shape[1] - frame_size)
        y = random.randint(0, inspected_image.shape[0] - frame_size)

        # Extract inspected frame
        inspected_frame = inspected_image[y:y+frame_size, x:x+frame_size]

        # Map frame to reference image using homography
        points = np.array([
            [x, y],
            [x + frame_size, y],
            [x + frame_size, y + frame_size],
            [x, y + frame_size]
        ], dtype=np.float32).reshape(-1, 1, 2)

        transformed_points = cv2.perspectiveTransform(points, H)

        # Calculate bounds of the transformed points
        x_min, y_min = np.int32(transformed_points.min(axis=0)[0])
        x_max, y_max = np.int32(transformed_points.max(axis=0)[0])

        # Clip the bounds to the size of the reference image
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(reference_image.shape[1], x_max)
        y_max = min(reference_image.shape[0], y_max)

        # Ensure the extracted region is not empty
        if x_min >= x_max or y_min >= y_max:
            continue

        # Extract and resize the reference frame
        reference_frame = reference_image[y_min:y_max, x_min:x_max]
        if reference_frame.size == 0:
            continue

        reference_frame_resized = cv2.resize(reference_frame, (frame_size, frame_size))
        sampled_frames.append((inspected_frame, reference_frame_resized))
    return sampled_frames

# Defect detection function
def detect_defects(inspected_image, reference_image, H, frame_size, threshold):
    """Detect defects by comparing inspected frames with reference frames."""
    binary_output = np.zeros_like(inspected_image, dtype=np.uint8)
    for y in range(0, inspected_image.shape[0] - frame_size, frame_size):
        for x in range(0, inspected_image.shape[1] - frame_size, frame_size):
            # Extract inspected frame
            inspected_frame = inspected_image[y:y+frame_size, x:x+frame_size]

            # Map frame to reference image using homography
            points = np.array([
                [x, y],
                [x + frame_size, y],
                [x + frame_size, y + frame_size],
                [x, y + frame_size]
            ], dtype=np.float32).reshape(-1, 1, 2)

            transformed_points = cv2.perspectiveTransform(points, H)
            x_min, y_min = np.int32(transformed_points.min(axis=0)[0])
            x_max, y_max = np.int32(transformed_points.max(axis=0)[0])

            # Clip the bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(reference_image.shape[1], x_max)
            y_max = min(reference_image.shape[0], y_max)

            # Ensure valid region
            if x_min >= x_max or y_min >= y_max:
                continue

            reference_frame = reference_image[y_min:y_max, x_min:x_max]
            if reference_frame.size == 0:
                continue

            reference_frame_resized = cv2.resize(reference_frame, (frame_size, frame_size))

            # Calculate similarity
            distance = calculate_frame_distance(inspected_frame, reference_frame_resized)

            # Mark defected frame
            if distance > threshold:
                binary_output[y:y+frame_size, x:x+frame_size] = 255
    return binary_output

# Parameters
num_samples = 500
frame_sizes = [20, 30, 50]  # Sizes of random frames

# Sampling frames and calculating threshold
sampled_frames = sample_frames(inspected_image, reference_image, H, num_samples, frame_sizes)
threshold = calculate_threshold(sampled_frames)

# Detecting defects
binary_defects = detect_defects(inspected_image, reference_image, H, frame_size, threshold)

# Interactive matching window
cv2.namedWindow("Inspected Image")
cv2.setMouseCallback("Inspected Image", move_frame)

# Initial display
move_frame(cv2.EVENT_MOUSEMOVE, frame_pos[0] + frame_size // 2, frame_pos[1] + frame_size // 2, None, None)

print("Move your mouse over the 'Inspected Image' window to control the frame.")
cv2.imshow("Reference Image", reference_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Plot defect detection results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Inspected Image")
plt.imshow(inspected_image, cmap='gray')
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Defected Pixels")
plt.imshow(binary_defects, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()