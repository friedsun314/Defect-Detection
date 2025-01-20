import cv2
import matplotlib.pyplot as plt
from graph_construct import construct_graph
from graph_comparison import compare_graphs, visualize_defects
from region_growing import region_growing_with_full_coverage
from region_merging import merge_similar_regions
from graph_comparison import compare_graphs, visualize_defects


def preprocess_image(image, method="gaussian", kernel_size=5):
    """
    Preprocess the image by smoothening the background.

    :param image: Grayscale input image (H x W).
    :param method: Smoothing method ('gaussian' or 'median').
    :param kernel_size: Size of the kernel to apply.
    :return: Smoothed image.
    """
    if method == "gaussian":
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif method == "median":
        return cv2.medianBlur(image, kernel_size)
    else:
        raise ValueError(f"Unsupported method: {method}. Use 'gaussian' or 'median'.")


if __name__ == "__main__":
    # Step 1: Load images
    inspected_image_path = "defective_examples/case1_inspected_image.tif"
    reference_image_path = "defective_examples/case1_reference_image.tif"

    inspected_image = cv2.imread(inspected_image_path, cv2.IMREAD_GRAYSCALE)
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)

    if inspected_image is None or reference_image is None:
        print("Error: Could not load one or both images.")
        exit(1)

    # Step 2: Preprocess images
    print("Preprocessing images...")
    inspected_smoothed = preprocess_image(inspected_image, method="gaussian", kernel_size=5)
    reference_smoothed = preprocess_image(reference_image, method="gaussian", kernel_size=5)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(inspected_smoothed, cmap="gray")
    plt.title("Inspected Image (Smoothed)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(reference_smoothed, cmap="gray")
    plt.title("Reference Image (Smoothed)")
    plt.axis("off")
    plt.show()

    # Step 3: Perform region growing and merging
    print("Segmenting images...")
    inspected_regions = region_growing_with_full_coverage(
        inspected_smoothed, n_samples=100, n_pixels=50, threshold=15.0
    )
    inspected_regions = merge_similar_regions(inspected_smoothed, inspected_regions, similarity_threshold=10.0)

    reference_regions = region_growing_with_full_coverage(
        reference_smoothed, n_samples=100, n_pixels=50, threshold=15.0
    )
    reference_regions = merge_similar_regions(reference_smoothed, reference_regions, similarity_threshold=10.0)

    # Step 4: Construct graphs
    print("Constructing graphs...")
    inspected_graph = construct_graph(inspected_smoothed, inspected_regions)
    reference_graph = construct_graph(reference_smoothed, reference_regions)

    print(f"Inspected Graph: {len(inspected_graph.nodes)} nodes, {len(inspected_graph.edges)} edges")
    print(f"Reference Graph: {len(reference_graph.nodes)} nodes, {len(reference_graph.edges)} edges")

    # Step 5: Compare graphs
    print("Comparing graphs to detect defects...")
    tolerances = {
        "size": 5,
        "intensity": 10,
        "centroid": 5,
    }
    defective_nodes = compare_graphs(inspected_graph, reference_graph, tolerances)

    print(f"Defective Nodes ({len(defective_nodes)}):")
    for node, data in defective_nodes:
        print(f"Node {node}: {data}")

    # Step 6: Visualize defects
    print("Visualizing defective regions...")
    visualize_defects(inspected_graph, defective_nodes, inspected_image.shape)