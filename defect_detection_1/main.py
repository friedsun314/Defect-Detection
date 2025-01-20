import cv2
import numpy as np
from matching import match_keypoints_good as match_keypoints, enable_interactive_frame
from defect_detection2 import detect_defects
from visualization import visualize_defect_map, visualize_some_defects

def crop_to_center(image, crop_size, H=None):
    """
    Crop the image to the center with the specified size. If homography (H) is provided,
    map the crop to the reference frame.

    Args:
        image (np.ndarray): Input grayscale image.
        crop_size (int): Desired size of the square crop.
        H (np.ndarray): Homography matrix. If None, crop directly from the center.

    Returns:
        np.ndarray: Cropped image.
    """
    h, w = image.shape
    cx, cy = w // 2, h // 2  # Center coordinates

    if H is not None:
        # Map center coordinates using homography
        center_point = np.array([[cx, cy]], dtype=np.float32).reshape(-1, 1, 2)
        mapped_point = cv2.perspectiveTransform(center_point, H)
        cx, cy = int(mapped_point[0][0][0]), int(mapped_point[0][0][1])

    x_start = max(0, cx - crop_size // 2)
    y_start = max(0, cy - crop_size // 2)
    x_end = min(w, cx + crop_size // 2)
    y_end = min(h, cy + crop_size // 2)
    return image[y_start:y_end, x_start:x_end]

def main():
    # Hardcoded paths to reference and inspected images
    reference_image_path = "defective_examples/case1_reference_image.tif"
    inspected_image_path = "defective_examples/case1_inspected_image.tif"
    
    # Load images
    image_ref = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    image_insp = cv2.imread(inspected_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Debugging: Print warnings if images are not loaded
    if image_ref is None:
        print(f"Warning: Unable to load reference image from {reference_image_path}")
    if image_insp is None:
        print(f"Warning: Unable to load inspected image from {inspected_image_path}")
    
    # Ensure images are loaded
    if image_ref is None or image_insp is None:
        raise FileNotFoundError("One or both image paths are incorrect. Please check the paths and try again.")
    
    print("Images loaded successfully.")
    
    # Step 1: Match keypoints and compute homography
    H, good_matches, keypoints_ref, keypoints_ins = match_keypoints(image_ref, image_insp, min_match_count=10, visualize=True)
    if H is None:
        raise ValueError("Homography could not be computed due to insufficient matches.")
    
    # Step 2: Enable interactive frame exploration
    print("Starting interactive frame exploration. Close the window to proceed.")
    enable_interactive_frame(image_ref, image_insp, H, frame_size=50)
    
    # Step 3: Crop to center after homography calculation
    crop_size = 200  # Example crop size (200x200 pixels)
    roi_ref = crop_to_center(image_ref, crop_size, H=None)
    roi_ins = crop_to_center(image_insp, crop_size, H=H)
    
    print("Center ROI extracted successfully after homography.")
    
    # Step 4: Detect defects using the homography matrix
    defect_map = detect_defects(
        image_ref=roi_ref,
        image_insp=roi_ins,
        H=H,
        max_frame_size_ratio=0.05,  # Adjusted to 5%
        distance_threshold=0.05
    )
    
    # Step 5: Visualize the defect map
    visualize_defect_map(roi_ins, defect_map, title="Defect Map")
    
    # Step 6: Visualize a subset of detected defects
    visualize_some_defects(roi_ins, defect_map, num_defects=10, title="Sample Defects")
    
    # Step 7: Save the defect map
    cv2.imwrite("defect_map.png", defect_map * 255)  # Scale to [0,255] for visibility
    print("Defect map saved as 'defect_map.png'.")

if __name__ == "__main__":
    main()