import numpy as np
import cv2
from skimage.feature import peak_local_max
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from matching import match_keypoints
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_corresponding_frame_coords(x_ins, y_ins, H, k):
    """Map inspected coordinates to reference coordinates using homography."""
    point_ins = np.array([[x_ins, y_ins]], dtype=np.float32).reshape(-1, 1, 2)
    point_ref = cv2.perspectiveTransform(point_ins, H)
    x_ref, y_ref = point_ref[0][0]
    x_ref = int(round(x_ref)) - k // 2
    y_ref = int(round(y_ref)) - k // 2
    return x_ref, y_ref


def detect_local_maxima_minima(image):
    """Detect local maxima and minima in the image."""
    maxima_coords = peak_local_max(image, footprint=np.ones((3, 3)), exclude_border=False)
    minima_coords = peak_local_max(-image, footprint=np.ones((3, 3)), exclude_border=False)

    # Convert coordinates to binary masks
    maxima_mask = np.zeros_like(image, dtype=bool)
    minima_mask = np.zeros_like(image, dtype=bool)
    maxima_mask[tuple(maxima_coords.T)] = True
    minima_mask[tuple(minima_coords.T)] = True

    return maxima_mask, minima_mask


def compute_distance(frame_ref, frame_ins):
    """Compute the distance between two frames using SSIM."""
    min_side = min(frame_ref.shape)
    win_size = 7 if min_side >= 7 else min_side  # Adjust win_size for small frames

    try:
        distance = 1 - ssim(frame_ref, frame_ins, data_range=frame_ref.max() - frame_ref.min(), win_size=win_size)
    except ValueError as e:
        logging.error(f"SSIM computation failed: {e}")
        return None
    return distance


def perform_expensive_matching(x, y, image_ref, image_insp, H, k):
    """Perform expensive matching for a single pixel using local and global homographies."""
    x_ref, y_ref = get_corresponding_frame_coords(x, y, H, k)
    if not (0 <= x_ref < image_ref.shape[1] - k and 0 <= y_ref < image_ref.shape[0] - k):
        return None, None

    frame_ins = image_insp[y:y + k, x:x + k]
    frame_ref = image_ref[y_ref:y_ref + k, x_ref:x_ref + k]

    if frame_ins.shape != (k, k) or frame_ref.shape != (k, k):
        return None, None

    distance = compute_distance(frame_ref, frame_ins)
    return distance, (frame_ref, frame_ins)


def region_growth(defect_map, image_insp, threshold=0.5):
    """Perform region growth to detect more defected pixels."""
    h, w = defect_map.shape
    visited = np.zeros_like(defect_map, dtype=bool)
    growth = np.copy(defect_map)

    for y in range(h):
        for x in range(w):
            if growth[y, x] == 1 and not visited[y, x]:
                stack = [(y, x)]
                while stack:
                    cy, cx = stack.pop()
                    if visited[cy, cx]:
                        continue
                    visited[cy, cx] = True
                    for ny, nx in [(cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)]:
                        if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                            local_diff = abs(float(image_insp[cy, cx]) - float(image_insp[ny, nx]))
                            if local_diff > threshold:
                                continue
                            growth[ny, nx] = 1
                            stack.append((ny, nx))
    return growth


def detect_defects(image_ref, image_insp, H, k_values=[3, 5, 7], distance_threshold=1e-6):
    """Detect defects using local maxima, minima, matching, and region growth."""
    maxima, minima = detect_local_maxima_minima(image_insp)
    candidate_pixels = np.argwhere(maxima | minima)

    defect_map = np.zeros_like(image_insp, dtype=np.uint8)
    for (y, x) in candidate_pixels:
        logging.info(f"Processing pixel ({x}, {y})...")
        for k in k_values:
            distance, frames = perform_expensive_matching(x, y, image_ref, image_insp, H, k)
            if distance is None:
                continue

            logging.info(f"Distance for pixel ({x}, {y}) with k={k}: {distance:.8f}")
            if distance > distance_threshold:
                defect_map[y, x] = 1
                break

    defect_map = region_growth(defect_map, image_insp, threshold=0.5)
    return defect_map


def visualize_defect_map(defect_map, image_insp):
    """Visualize the defect map overlaid on the inspected image."""
    overlay = cv2.cvtColor(image_insp, cv2.COLOR_GRAY2BGR)
    overlay[defect_map == 1] = [0, 0, 255]

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Defect Map Visualization")
    plt.axis("off")
    plt.show()


def visualize_matching_quality(image_ref, image_insp, defect_map, candidate_pixels):
    """Visualize the matching quality for defected regions."""
    for (y, x) in candidate_pixels:
        if defect_map[y, x] == 1:
            plt.figure(figsize=(12, 6))

            # Reference patch
            plt.subplot(1, 2, 1)
            plt.imshow(image_ref[y - 3:y + 4, x - 3:x + 4], cmap="gray")
            plt.title("Reference Patch")
            plt.axis("off")

            # Inspected patch
            plt.subplot(1, 2, 2)
            plt.imshow(image_insp[y - 3:y + 4, x - 3:x + 4], cmap="gray")
            plt.title("Inspected Patch")
            plt.axis("off")

            plt.show()


def main():
    reference_image_path = "defective_examples/case1_reference_image.tif"
    inspected_image_path = "defective_examples/case1_inspected_image.tif"

    image_ref = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    image_insp = cv2.imread(inspected_image_path, cv2.IMREAD_GRAYSCALE)

    if image_ref is None or image_insp is None:
        raise FileNotFoundError("Ensure the reference and inspected images are correctly loaded.")

    H, _, _, _ = match_keypoints(image_ref, image_insp, min_match_count=10)
    if H is None:
        raise ValueError("Homography could not be computed.")

    defect_map = detect_defects(image_ref, image_insp, H)

    visualize_defect_map(defect_map, image_insp)

    # Visualize matching quality for defected regions
    maxima, minima = detect_local_maxima_minima(image_insp)
    candidate_pixels = np.argwhere(maxima | minima)
    visualize_matching_quality(image_ref, image_insp, defect_map, candidate_pixels)


if __name__ == "__main__":
    main()