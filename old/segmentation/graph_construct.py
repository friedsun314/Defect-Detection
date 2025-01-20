import numpy as np
import networkx as nx
from scipy.spatial import distance
import matplotlib.pyplot as plt
import cv2


def construct_graph(image, regions):
    """
    Construct a graph from segmented regions.

    :param image: Grayscale image (H x W).
    :param regions: List of regions, where each region is a set of (x, y) pixel coordinates.
    :return: A networkx graph with nodes and weighted edges.
    """
    G = nx.Graph()

    # Compute region properties
    region_properties = []
    for region in regions:
        pixels = list(region)
        size = len(pixels)
        mean_intensity = np.mean([image[y, x] for x, y in pixels])
        centroid = np.mean(pixels, axis=0)  # (mean_x, mean_y)
        region_properties.append((region, size, mean_intensity, tuple(centroid)))

    # Add nodes with attributes
    for i, (_, size, mean_intensity, centroid) in enumerate(region_properties):
        G.add_node(
            i,
            size=size,
            mean_intensity=mean_intensity,
            centroid=centroid,
        )

    # Add edges between neighboring regions
    for i, (region_i, _, intensity_i, centroid_i) in enumerate(region_properties):
        for j, (region_j, _, intensity_j, centroid_j) in enumerate(region_properties):
            if i >= j:
                continue  # Avoid duplicate edges

            # Check spatial proximity
            if are_regions_neighbors(region_i, region_j):
                # Compute edge attributes
                intensity_diff = abs(intensity_i - intensity_j)
                centroid_dist = distance.euclidean(centroid_i, centroid_j)
                shared_boundary = compute_shared_boundary(region_i, region_j)

                # Add edge
                G.add_edge(
                    i,
                    j,
                    weight=intensity_diff,
                    centroid_dist=centroid_dist,
                    shared_boundary=shared_boundary,
                )

    return G


def are_regions_neighbors(region1, region2, max_distance=1):
    """
    Check if two regions are neighbors by verifying if any pixel from one region is within a given distance
    of any pixel from the other region.

    :param region1: Set of (x, y) pixel coordinates.
    :param region2: Set of (x, y) pixel coordinates.
    :param max_distance: Maximum Manhattan distance to consider regions as neighbors.
    :return: True if regions are neighbors, False otherwise.
    """
    for x1, y1 in region1:
        for x2, y2 in region2:
            if abs(x1 - x2) <= max_distance and abs(y1 - y2) <= max_distance:
                return True
    return False


def compute_shared_boundary(region1, region2):
    """
    Compute the number of shared boundary pixels between two regions.

    :param region1: Set of (x, y) pixel coordinates.
    :param region2: Set of (x, y) pixel coordinates.
    :return: Number of shared boundary pixels.
    """
    return len(region1 & region2)


def visualize_graph(graph, title="Graph Visualization"):
    """
    Visualize the constructed graph using networkx.

    :param graph: The constructed graph.
    :param title: Title for the plot.
    """
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(graph)
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=[100 + data["size"] for _, data in graph.nodes(data=True)],
        node_color="lightblue",
        edge_color="gray",
    )
    labels = nx.get_edge_attributes(graph, "weight")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
    plt.title(title)
    plt.show()


# Embedded Tests
if __name__ == "__main__":
    # Test 1: Synthetic regions
    test_image = np.zeros((100, 100), dtype=np.uint8)
    region1 = {(x, 10) for x in range(10, 30)}
    region2 = {(x, 20) for x in range(20, 40)}
    region3 = {(x, 30) for x in range(30, 50)}

    regions = [region1, region2, region3]
    for region, intensity in zip(regions, [50, 100, 150]):
        for x, y in region:
            test_image[y, x] = intensity

    graph = construct_graph(test_image, regions)
    print("Synthetic Graph:")
    print(f"Nodes: {graph.nodes(data=True)}")
    print(f"Edges: {graph.edges(data=True)}")
    visualize_graph(graph, title="Synthetic Graph")

    # Test 2: Real image
    inspected_image_path = "defective_examples/case1_inspected_image.tif"
    inspected_image = cv2.imread(inspected_image_path, cv2.IMREAD_GRAYSCALE)

    if inspected_image is None:
        print("Error: Could not load the inspected image.")
        exit(1)

    # Assume 'regions' are already obtained via segmentation
    from region_growing import region_growing_with_full_coverage
    from region_merging import merge_similar_regions

    # Step 1: Perform region growing and merging
    regions = region_growing_with_full_coverage(inspected_image, n_samples=1000, n_pixels=50, threshold=22.0)
    merged_regions = merge_similar_regions(inspected_image, regions, similarity_threshold=10.0)

    # Step 2: Construct graph
    graph = construct_graph(inspected_image, merged_regions)
    print("Real Image Graph:")
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")
    visualize_graph(graph, title="Real Image Graph")