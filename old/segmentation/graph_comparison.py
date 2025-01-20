import numpy as np


def compare_graphs(inspected_graph, reference_graph, tolerances):
    """
    Compare two graphs and identify nodes in the inspected graph without matching nodes in the reference graph.

    :param inspected_graph: NetworkX graph of the inspected image.
    :param reference_graph: NetworkX graph of the reference image.
    :param tolerances: Dictionary with tolerances for matching nodes:
        - 'size': Absolute difference in size.
        - 'intensity': Absolute difference in mean intensity.
        - 'centroid': Maximum Euclidean distance between centroids.
    :return: List of defective nodes from the inspected graph.
    """
    defective_nodes = []

    for inspected_node, inspected_data in inspected_graph.nodes(data=True):
        is_matched = False

        for reference_node, reference_data in reference_graph.nodes(data=True):
            # Check size
            size_diff = abs(inspected_data["size"] - reference_data["size"])
            if size_diff > tolerances["size"]:
                continue

            # Check mean intensity
            intensity_diff = abs(inspected_data["mean_intensity"] - reference_data["mean_intensity"])
            if intensity_diff > tolerances["intensity"]:
                continue

            # Check centroid distance
            centroid_dist = np.linalg.norm(
                np.array(inspected_data["centroid"]) - np.array(reference_data["centroid"])
            )
            if centroid_dist > tolerances["centroid"]:
                continue

            # If all criteria are met, the node is matched
            is_matched = True
            break

        # If no match was found, declare the node as defective
        if not is_matched:
            defective_nodes.append((inspected_node, inspected_data))

    return defective_nodes


def visualize_defects(inspected_graph, defective_nodes, image_shape):
    """
    Visualize defective regions on the inspected image.

    :param inspected_graph: NetworkX graph of the inspected image.
    :param defective_nodes: List of defective nodes and their attributes.
    :param image_shape: Shape of the original image (H, W).
    """
    defect_image = np.zeros(image_shape, dtype=np.uint8)

    for node, data in defective_nodes:
        size = data["size"]
        centroid = tuple(map(int, data["centroid"]))
        defect_image[
            max(0, centroid[1] - size // 2): min(image_shape[0], centroid[1] + size // 2),
            max(0, centroid[0] - size // 2): min(image_shape[1], centroid[0] + size // 2),
        ] = 255

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))
    plt.imshow(defect_image, cmap="gray")
    plt.title("Defective Regions")
    plt.axis("off")
    plt.show()


# Embedded Tests
if __name__ == "__main__":
    import networkx as nx

    # Define tolerances
    tolerances = {
        "size": 5,
        "intensity": 10,
        "centroid": 5,
    }

    # Test 1: Single defective node
    inspected_graph = nx.Graph()
    inspected_graph.add_node(0, size=20, mean_intensity=100, centroid=(30, 30))
    inspected_graph.add_node(1, size=25, mean_intensity=150, centroid=(60, 60))
    inspected_graph.add_node(2, size=15, mean_intensity=200, centroid=(90, 90))

    reference_graph = nx.Graph()
    reference_graph.add_node(0, size=20, mean_intensity=100, centroid=(30, 30))
    reference_graph.add_node(1, size=25, mean_intensity=150, centroid=(60, 60))

    defective_nodes = compare_graphs(inspected_graph, reference_graph, tolerances)
    print("\nTest 1 - Defective Nodes:")
    for node, data in defective_nodes:
        print(f"Node {node}: {data}")

    visualize_defects(inspected_graph, defective_nodes, (100, 100))

    # Test 2: No defects
    inspected_graph = nx.Graph()
    inspected_graph.add_node(0, size=20, mean_intensity=100, centroid=(30, 30))
    inspected_graph.add_node(1, size=25, mean_intensity=150, centroid=(60, 60))

    reference_graph = nx.Graph()
    reference_graph.add_node(0, size=20, mean_intensity=100, centroid=(30, 30))
    reference_graph.add_node(1, size=25, mean_intensity=150, centroid=(60, 60))

    defective_nodes = compare_graphs(inspected_graph, reference_graph, tolerances)
    print("\nTest 2 - Defective Nodes (Expected: None):")
    print(defective_nodes)  # Should be empty

    # Test 3: Tolerance check
    inspected_graph = nx.Graph()
    inspected_graph.add_node(0, size=20, mean_intensity=100, centroid=(30, 30))
    inspected_graph.add_node(1, size=25, mean_intensity=150, centroid=(60, 60))
    inspected_graph.add_node(2, size=22, mean_intensity=105, centroid=(31, 30))  # Within tolerance

    reference_graph = nx.Graph()
    reference_graph.add_node(0, size=20, mean_intensity=100, centroid=(30, 30))
    reference_graph.add_node(1, size=25, mean_intensity=150, centroid=(60, 60))

    defective_nodes = compare_graphs(inspected_graph, reference_graph, tolerances)
    print("\nTest 3 - Defective Nodes (Expected: None due to tolerance):")
    print(defective_nodes)  # Should be empty due to tolerances

    # Test 4: Realistic large differences
    inspected_graph = nx.Graph()
    inspected_graph.add_node(0, size=20, mean_intensity=100, centroid=(30, 30))
    inspected_graph.add_node(1, size=25, mean_intensity=150, centroid=(60, 60))
    inspected_graph.add_node(2, size=40, mean_intensity=200, centroid=(90, 90))  # Large differences

    reference_graph = nx.Graph()
    reference_graph.add_node(0, size=20, mean_intensity=100, centroid=(30, 30))
    reference_graph.add_node(1, size=25, mean_intensity=150, centroid=(60, 60))

    defective_nodes = compare_graphs(inspected_graph, reference_graph, tolerances)
    print("\nTest 4 - Defective Nodes (Expected: Node 2):")
    for node, data in defective_nodes:
        print(f"Node {node}: {data}")

    visualize_defects(inspected_graph, defective_nodes, (100, 100))