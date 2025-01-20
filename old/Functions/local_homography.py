import cv2
import numpy as np
import matplotlib.pyplot as plt

def divide_into_patches(image, patch_size):
    """Divide the image into overlapping patches of given size."""
    h, w = image.shape
    patches = []
    step = patch_size // 2  # Overlapping step
    for y in range(0, h - patch_size + 1, step):
        for x in range(0, w - patch_size + 1, step):
            patches.append((x, y, patch_size, patch_size))
    return patches

def compute_local_homography(img1, img2, patch_size):
    """Perform local homography matching between two images."""
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Extract matched keypoints
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches])
    
    patches = divide_into_patches(img1, patch_size)
    result_img = img2.copy()
    
    for x, y, w, h in patches:
        # Filter keypoints inside the current patch
        patch_mask = (
            (src_pts[:, 0] >= x) & (src_pts[:, 0] < x + w) &
            (src_pts[:, 1] >= y) & (src_pts[:, 1] < y + h)
        )
        
        src_patch_pts = src_pts[patch_mask]
        dst_patch_pts = dst_pts[patch_mask]
        
        if len(src_patch_pts) >= 4:  # Minimum 4 points for homography
            H, mask = cv2.findHomography(src_patch_pts, dst_patch_pts, cv2.RANSAC, 5.0)
            if H is not None:
                # Apply the local homography to this patch
                patch_corners = np.float32([
                    [x, y],
                    [x + w, y],
                    [x + w, y + h],
                    [x, y + h]
                ]).reshape(-1, 1, 2)
                
                warped_corners = cv2.perspectiveTransform(patch_corners, H)
                warped_corners = warped_corners.astype(int)
                
                # Draw the transformed region on the result image
                cv2.polylines(result_img, [warped_corners], isClosed=True, color=(0, 255, 0), thickness=2)
    
    return result_img

def visualize_results(img1, img2, result_img):
    """Visualize original images and the local homography result."""
    plt.figure(figsize=(15, 10))

    plt.subplot(1, 3, 1)
    plt.title("Image 1 (Reference)")
    plt.imshow(img1, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Image 2 (Inspected)")
    plt.imshow(img2, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Local Homography Result")
    plt.imshow(result_img, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Paths to your images
image1_path = "data/defective_examples/case1_inspected_image.tif"
image2_path = "data/defective_examples/case1_reference_image.tif"

# Load images in grayscale
img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

if img1 is not None and img2 is not None:
    patch_size = 100  # Size of each patch (adjust as needed)
    result = compute_local_homography(img1, img2, patch_size)
    visualize_results(img1, img2, result)
else:
    print("Error: One or both images could not be loaded.")