import cv2
import numpy as np
import matplotlib.pyplot as plt

def sift_image_alignment(reference_path, inspected_path):
    """
    Perform image alignment using SIFT and plot the aligned result.
    
    Parameters:
        reference_path (str): Path to the reference image.
        inspected_path (str): Path to the inspected image.
    """
    # Load images
    reference = cv2.imread(reference_path)
    inspected = cv2.imread(inspected_path)

    # Convert to grayscale
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    input_gray = cv2.cvtColor(inspected, cv2.COLOR_BGR2GRAY)

    # Detect SIFT keypoints and descriptors
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(ref_gray, None)
    kp2, des2 = sift.detectAndCompute(input_gray, None)

    # Match descriptors using BFMatcher
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract keypoint coordinates
    points1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Estimate the affine transformation matrix
    matrix, _ = cv2.estimateAffine2D(points2, points1, method=cv2.RANSAC)

    # Align the inspected image to the reference image
    aligned_image = cv2.warpAffine(inspected, matrix, (reference.shape[1], reference.shape[0]))

    # Plot results
    plt.figure(figsize=(12, 6))
    
    # Original images
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB))
    plt.title("Reference Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(inspected, cv2.COLOR_BGR2RGB))
    plt.title("Inspected Image")
    plt.axis("off")
    
    # Aligned image over reference
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB), alpha=0.5, label='Reference')
    plt.imshow(cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB), alpha=0.5, label='Aligned')
    plt.title("Aligned Image Over Reference")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

# Paths to your images
reference_path = "data/defective_examples/case1_reference_image.tif"
inspected_path = "data/defective_examples/case1_inspected_image.tif"

# Align and plot
sift_image_alignment(reference_path, inspected_path)