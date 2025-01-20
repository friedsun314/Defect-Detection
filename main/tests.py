import numpy as np
import cv2
import matplotlib.pyplot as plt
import subprocess
import os


def generate_pattern_image(shape, pattern_type="stripes", frequency=10):
    h, w = shape
    image = np.zeros((h, w), dtype=np.uint8)

    if pattern_type == "stripes":
        for y in range(0, h, frequency):
            image[y:y + frequency // 2, :] = 255
    elif pattern_type == "checkerboard":
        for y in range(0, h, frequency):
            for x in range(0, w, frequency):
                if (y // frequency + x // frequency) % 2 == 0:
                    image[y:y + frequency // 2, x:x + frequency // 2] = 255
    return image


def add_defects(image, defect_type="spots", defect_positions=None, intensity=50, size=5):
    image_with_defects = image.copy()

    if defect_type == "spots" and defect_positions:
        for pos in defect_positions:
            y, x = pos
            cv2.circle(image_with_defects, (x, y), size, intensity, -1)

    return image_with_defects


def visualize_test_images(reference_image, inspected_image, title="Test Images"):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Reference Image")
    plt.imshow(reference_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Inspected Image with Defects")
    plt.imshow(inspected_image, cmap="gray")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def save_test_images(ref_image, inspected_image, case_name):
    """
    Saves the reference and inspected images to the appropriate paths.

    Args:
        ref_image (numpy.ndarray): Reference image.
        inspected_image (numpy.ndarray): Inspected image with defects.
        case_name (str): Name of the test case.
    """
    os.makedirs("data/defective_examples", exist_ok=True)
    ref_path = f"data/defective_examples/{case_name}_reference_image.tif"
    inspected_path = f"data/defective_examples/{case_name}_inspected_image.tif"

    cv2.imwrite(ref_path, ref_image)
    print(f"Saved reference image for {case_name} at {ref_path}")

    cv2.imwrite(inspected_path, inspected_image)
    print(f"Saved inspected image for {case_name} at {inspected_path}")

    return ref_path, inspected_path


def run_pipeline(reference_image_path, inspected_image_path):
    """
    Runs the pipeline by executing main.py as a subprocess.

    Args:
        reference_image_path (str): Path to the reference image.
        inspected_image_path (str): Path to the inspected image.
    """
    print("Running pipeline...")
    result = subprocess.run(
        ["python", "main.py", reference_image_path, inspected_image_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("Error running pipeline:")
        print(result.stderr)


if __name__ == "__main__":
    # # Example 1: Horizontal Stripes with Defects
    # ref_image = generate_pattern_image((256, 256), pattern_type="stripes", frequency=10)
    # inspected_image = add_defects(ref_image, defect_type="spots", defect_positions=[(50, 50), (128, 128)], intensity=100, size=5)
    # visualize_test_images(ref_image, inspected_image, title="Example 1: Stripes with Spots")

    # case1_ref_path, case1_ins_path = save_test_images(ref_image, inspected_image, "case1")
    # run_pipeline(case1_ref_path, case1_ins_path)

    # # Example 2: Checkerboard with Line Defects
    # ref_image = generate_pattern_image((256, 256), pattern_type="checkerboard", frequency=20)
    # inspected_image = add_defects(ref_image, defect_type="lines", defect_positions=[(60, 60), (150, 100)], intensity=150, size=2)
    # visualize_test_images(ref_image, inspected_image, title="Example 2: Checkerboard with Lines")

    # case2_ref_path, case2_ins_path = save_test_images(ref_image, inspected_image, "case2")
    # run_pipeline(case2_ref_path, case2_ins_path)

    # Example 3: Actual Images
    actual_ref_path = "data/defective_examples/case1_reference_image.tif"
    actual_ins_path = "data/defective_examples/case1_inspected_image.tif"

    if os.path.exists(actual_ref_path) and os.path.exists(actual_ins_path):
        print("\nRunning pipeline on actual images...")
        run_pipeline(actual_ref_path, actual_ins_path)
    else:
        print(f"\nActual images not found at {actual_ref_path} and {actual_ins_path}. Skipping...")