import cv2
import numpy as np
from skimage.segmentation import felzenszwalb
from skimage.color import label2rgb
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
from PIL import Image


def felzenszwalb_segmentation(image, scale=100, sigma=0.8, min_size=50):
    """
    Segment the image into equivalence classes using Felzenszwalb's method.
    """
    return felzenszwalb(image, scale=scale, sigma=sigma, min_size=min_size)


def compute_persistent_homology(segmentation_map):
    """
    Compute persistent homology for the segmentation map.
    """
    # Convert segmentation map into a distance matrix
    distance_matrix = segmentation_map.astype(np.float32)
    # Compute persistent homology using Ripser
    diagrams = ripser(distance_matrix, maxdim=1)['dgms']
    return diagrams


def compare_topologies(reference_diagrams, target_diagrams):
    """
    Compare the persistence diagrams of the reference and target images.
    """
    # Plot the diagrams for visual inspection
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    plot_diagrams(reference_diagrams, ax=axs[0], title="Reference Topology")
    plot_diagrams(target_diagrams, ax=axs[1], title="Target Topology")
    plt.show()

    # Compute a topological similarity score (e.g., bottleneck distance)
    from persim import bottleneck
    bottleneck_distance = bottleneck(reference_diagrams[1], target_diagrams[1])
    print(f"Bottleneck Distance (H1): {bottleneck_distance}")

    return bottleneck_distance


def highlight_defects(reference, target, reference_diagrams, target_diagrams):
    """
    Highlight defects by identifying topological mismatches.
    """
    # Compute a binary difference map based on segmentation
    difference = np.abs(reference - target).astype(np.uint8)
    difference_map = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)[1]

    # Visualize the difference
    plt.imshow(difference_map, cmap='hot')
    plt.title("Topological Defects")
    plt.axis('off')
    plt.show()

    return difference_map


# Main workflow
if __name__ == "__main__":
    # Paths to reference and target images
    reference_path = "defective_examples/case1_reference_image.tif"
    target_path = "defective_examples/case1_inspected_image.tif"

    # Load images as grayscale for segmentation
    reference = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

    # Ensure both images have the same dimensions
    target_resized = cv2.resize(target, (reference.shape[1], reference.shape[0]))

    # Step 1: Segment both images into equivalence classes
    reference_segmentation = felzenszwalb_segmentation(reference, scale=100, sigma=0.8, min_size=50)
    target_segmentation = felzenszwalb_segmentation(target_resized, scale=100, sigma=0.8, min_size=50)

    # Step 2: Compute persistent homology for both segmentations
    reference_diagrams = compute_persistent_homology(reference_segmentation)
    target_diagrams = compute_persistent_homology(target_segmentation)

    # Step 3: Compare topologies
    bottleneck_distance = compare_topologies(reference_diagrams, target_diagrams)

    # Step 4: Highlight defects
    defect_map = highlight_defects(reference_segmentation, target_segmentation,
                                   reference_diagrams, target_diagrams)

    # Save the defect map for further inspection
    defect_map_image = Image.fromarray(defect_map)
    defect_map_image.show(title="Defect Map")