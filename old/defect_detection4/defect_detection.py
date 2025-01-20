import cv2
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import normalize_images
from matching import perform_matching, extract_reference_frame

def initialize_defect_map(image_shape):
    defect_map = np.full(image_shape, 255, dtype=np.uint8)
    plt.figure(figsize=(6, 6))
    plt.title("Initialized Defect Map (All Defective)")
    plt.imshow(defect_map, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    return defect_map

def detect_defects(
    inspected_image, reference_image, H, frame_sizes, big_frame_size, similarity_threshold=0.8
):
    defect_map = initialize_defect_map(inspected_image.shape)

    for frame_size in frame_sizes:
        print(f"Processing frame size: {frame_size}x{frame_size}")

        for y in range(0, inspected_image.shape[0] - frame_size, frame_size):
            for x in range(0, inspected_image.shape[1] - frame_size, frame_size):
                inspected_frame = inspected_image[y:y+frame_size, x:x+frame_size]

                big_x = max(0, x + frame_size // 2 - big_frame_size // 2)
                big_y = max(0, y + frame_size // 2 - big_frame_size // 2)
                big_x_end = min(inspected_image.shape[1], big_x + big_frame_size)
                big_y_end = min(inspected_image.shape[0], big_y + big_frame_size)
                big_frame = (big_x, big_y, big_x_end - big_x, big_y_end - big_y)

                reference_region = extract_reference_frame(reference_image, H, big_frame)

                if reference_region.shape[0] < frame_size or reference_region.shape[1] < frame_size:
                    continue

                result = cv2.matchTemplate(reference_region, inspected_frame, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(reference_region, cmap="gray")
                rect = plt.Rectangle(max_loc, frame_size, frame_size, edgecolor="yellow", fill=False, lw=2)
                ax.add_patch(rect)
                ax.set_title(f"Best Match Location (Score: {max_val:.2f})")
                ax.axis("off")
                plt.tight_layout()
                plt.show()

                if max_val >= similarity_threshold:
                    defect_map[y:y+frame_size, x:x+frame_size] = 0

    return defect_map

if __name__ == "__main__":
    reference_image_path = "data/defective_examples/case1_reference_image.tif"
    inspected_image_path = "data/defective_examples/case1_inspected_image.tif"

    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    inspected_image = cv2.imread(inspected_image_path, cv2.IMREAD_GRAYSCALE)

    if reference_image is None or inspected_image is None:
        raise FileNotFoundError("One or both image paths are incorrect.")

    norm_inspected_image, norm_reference_image = normalize_images(inspected_image, reference_image)

    H, _, _, _, _ = perform_matching(norm_reference_image, norm_inspected_image)

    frame_sizes = [15, 25, 35]
    big_frame_size = 100
    similarity_threshold = 0.8

    defect_map = detect_defects(
        norm_inspected_image, norm_reference_image, H, frame_sizes, big_frame_size, similarity_threshold
    )

    plt.figure(figsize=(10, 5))
    plt.title("Final Defect Map")
    plt.imshow(defect_map, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.show()