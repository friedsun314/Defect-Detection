import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import logging
import cv2
from multiprocessing import Pool, cpu_count
from functools import partial

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def _find_best_match(args):
    inspected_frame, ref_image = args
    res = cv2.matchTemplate(ref_image, inspected_frame, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return max_loc, max_val

def align_images_majority_vote(
    ref_image,
    inspected_image,
    n_frames=50,
    top_percent=5,
    frame_size_range=(20, 100),
    debug=False,
    visualize=False
):
    """
    Align the inspected image to the reference image using a robust, frame-based approach.
    Includes optional visualizations of random frames and their matches.

    Parameters:
        ref_image (np.ndarray): The reference image (2D array).
        inspected_image (np.ndarray): The inspected image (2D array).
        n_frames (int): Number of random frames to sample from the inspected image.
        top_percent (float): Percentage of top pairs to consider for voting.
        frame_size_range (tuple): Min and max frame sizes (height, width).
        debug (bool): If True, enable detailed logging.
        visualize (bool): If True, display visualizations for each frame.

    Returns:
        best_shift (tuple): The shift (y_shift, x_shift) determined by majority vote.
    """
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    ref_h, ref_w = ref_image.shape
    insp_h, insp_w = inspected_image.shape
    all_pairs = []

    logging.info(f"Reference Image Shape: {ref_image.shape}")
    logging.info(f"Inspected Image Shape: {inspected_image.shape}\n")

    min_frame_size, max_frame_size = frame_size_range

    # Prepare frames
    frames = []
    frame_coords = []
    for frame_idx in range(n_frames):
        # Randomly determine frame size with constraints
        frame_h = np.random.randint(min_frame_size, min(max_frame_size, insp_h // 2))
        frame_w = np.random.randint(min_frame_size, min(max_frame_size, insp_w // 2))

        # Ensure the frame fits within the inspected image
        if insp_h < frame_h or insp_w < frame_w:
            logging.debug(f"Skipping frame {frame_idx + 1} due to size constraints.")
            continue

        start_y = np.random.randint(0, insp_h - frame_h + 1)
        start_x = np.random.randint(0, insp_w - frame_w + 1)
        inspected_frame = inspected_image[start_y:start_y + frame_h, start_x:start_x + frame_w]

        frames.append(inspected_frame)
        frame_coords.append((start_y, start_x, frame_h, frame_w))

        logging.debug(f"Frame {frame_idx + 1}: Size=({frame_h}, {frame_w}), "
                      f"Coordinates=({start_y}, {start_x})")

    # Prepare for parallel processing
    pool = Pool(processes=cpu_count())
    results = []
    for idx, (inspected_frame, (start_y, start_x, frame_h, frame_w)) in enumerate(zip(frames, frame_coords)):
        logging.debug(f"Processing Frame {idx + 1}")
        # Extract corresponding region from reference image
        ref_region = ref_image
        args = (inspected_frame, ref_region)
        results.append(pool.apply_async(_find_best_match, args=(args,)))

    pool.close()
    pool.join()

    for idx, res in enumerate(results):
        (ref_x, ref_y), score = res.get()
        start_y, start_x, frame_h, frame_w = frame_coords[idx]

        # Calculate shift
        y_shift = ref_y - start_y
        x_shift = ref_x - start_x
        all_pairs.append(((start_y, start_x), (ref_y, ref_x), score, (y_shift, x_shift)))

        logging.debug(f"Frame {idx + 1}: Ref Coord=({ref_y}, {ref_x}), Score={score:.4f}, "
                      f"Shift=({y_shift}, {x_shift})")

        if visualize:
            best_match_frame = ref_image[ref_y:ref_y + frame_h, ref_x:ref_x + frame_w]

            plt.figure(figsize=(12, 4))
            plt.suptitle(f"Frame {idx + 1}: Inspected vs. Best Match", fontsize=14)

            # Inspected Frame
            plt.subplot(1, 3, 1)
            plt.title("Inspected Frame")
            plt.imshow(inspected_frame, cmap="gray")
            plt.axis("off")

            # Best-Matching Frame
            plt.subplot(1, 3, 2)
            plt.title("Best-Matching Frame (Reference)")
            plt.imshow(best_match_frame, cmap="gray")
            plt.axis("off")

            # Overlay
            plt.subplot(1, 3, 3)
            plt.title("Overlay")
            plt.imshow(inspected_frame, cmap="gray", alpha=0.5)
            plt.imshow(best_match_frame, cmap="jet", alpha=0.5)
            plt.axis("off")

            plt.show()

    # Sort all pairs by score (higher is better for TM_CCOEFF_NORMED)
    all_pairs.sort(key=lambda x: x[2], reverse=True)  # Sort by score descending
    top_k = max(1, int(len(all_pairs) * top_percent / 100))
    top_pairs = all_pairs[:top_k]

    logging.info("\nTop {}% Pairs:".format(top_percent))
    for (start_y, start_x), (ref_y, ref_x), score, shift in top_pairs:
        logging.info(f"  Inspected Coord: ({start_y}, {start_x}), "
                     f"Ref Coord: ({ref_y}, {ref_x}), Score: {score:.4f}, Shift: {shift}")

    # Calculate suggested shifts from the top pairs
    suggested_shifts = [shift for _, _, _, shift in top_pairs]

    logging.info("\nSuggested Shifts:")
    logging.info(suggested_shifts)
    best_shift = Counter(suggested_shifts).most_common(1)[0][0]
    logging.info(f"Majority Vote: Best Shift = {best_shift}")
    return best_shift


# Load grayscale images
ref_image = cv2.imread('defective_examples/case1_reference_image.tif', cv2.IMREAD_GRAYSCALE)
inspected_image = cv2.imread('defective_examples/case1_inspected_image.tif', cv2.IMREAD_GRAYSCALE)

# Align images
best_shift = align_images_majority_vote(
    ref_image,
    inspected_image,
    n_frames=100,
    top_percent=5,
    frame_size_range=(30, 80),
    debug=True,
    visualize=True
)

print(f"Determined Best Shift: {best_shift}")