import igraph as ig
import numpy as np

def build_rag(segmentation):
    """
    Build a Region Adjacency Graph (RAG) from a segmented frame using igraph.
    
    Args:
        segmentation (np.ndarray): Segmented image with integer labels.
    
    Returns:
        igraph.Graph: The constructed RAG with vertex 'name' attributes set to labels.
    """
    # Validate input
    if not isinstance(segmentation, np.ndarray):
        raise ValueError("Segmentation must be a numpy array.")
    if segmentation.ndim != 2:
        raise ValueError("Segmentation must be a 2D array.")
    if not np.issubdtype(segmentation.dtype, np.integer):
        raise ValueError("Segmentation labels must be integers.")
    
    labels = np.unique(segmentation)
    rag = ig.Graph(directed=False)
    rag.add_vertices(len(labels))  # Add vertices based on the number of unique labels
    
    # Create a mapping from label to vertex ID
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    
    # Assign 'name' attribute to each vertex for easy reference
    rag.vs["name"] = labels.tolist()
    
    h, w = segmentation.shape
    edges = set()
    for y in range(h):
        for x in range(w):
            current_label = segmentation[y, x]
            # Check right neighbor
            if x < w - 1:
                neighbor_label = segmentation[y, x+1]
                if neighbor_label != current_label:
                    edge = tuple(sorted((label_to_id[current_label], label_to_id[neighbor_label])))
                    edges.add(edge)
            # Check bottom neighbor
            if y < h - 1:
                neighbor_label = segmentation[y+1, x]
                if neighbor_label != current_label:
                    edge = tuple(sorted((label_to_id[current_label], label_to_id[neighbor_label])))
                    edges.add(edge)
    if edges:
        rag.add_edges(list(edges))
    return rag

#############################
# Testing Section
#############################

def test_build_rag():
    """
    Test the build_rag function with various cases.
    """
    # Test Case 1: Single Region
    print("Running Test Case 1: Single Region")
    seg_single = np.array([
        [1, 1],
        [1, 1]
    ])
    rag_single = build_rag(seg_single)
    assert rag_single.vcount() == 1, f"Expected 1 vertex, got {rag_single.vcount()}"
    assert rag_single.ecount() == 0, f"Expected 0 edges, got {rag_single.ecount()}"
    assert rag_single.vs["name"] == [1], f"Expected vertex name [1], got {rag_single.vs['name']}"
    print("Test Case 1 Passed: Single Region")
    
    # Test Case 2: Two Adjacent Regions
    print("Running Test Case 2: Two Adjacent Regions")
    seg_two = np.array([
        [1, 1, 2],
        [1, 1, 2],
        [3, 3, 2]
    ])
    rag_two = build_rag(seg_two)
    assert rag_two.vcount() == 3, f"Expected 3 vertices, got {rag_two.vcount()}"
    assert rag_two.ecount() == 3, f"Expected 3 edges, got {rag_two.ecount()}"
    expected_edges = {(1, 2), (1, 3), (2, 3)}
    edge_labels = set()
    for edge in rag_two.get_edgelist():
        edge_labels.add(tuple(sorted((rag_two.vs[edge[0]]["name"], rag_two.vs[edge[1]]["name"]))))
    assert edge_labels == expected_edges, f"Expected edges {expected_edges}, got {edge_labels}"
    print("Test Case 2 Passed: Two Adjacent Regions")
    
    # Test Case 3: Non-Adjacent Same Labels
    print("Running Test Case 3: Non-Adjacent Same Labels")
    seg_non_adjacent = np.array([
        [1, 2, 1],
        [2, 1, 2],
        [1, 2, 1]
    ])
    rag_non_adjacent = build_rag(seg_non_adjacent)
    assert rag_non_adjacent.vcount() == 2, f"Expected 2 vertices, got {rag_non_adjacent.vcount()}"
    assert rag_non_adjacent.ecount() == 1, f"Expected 1 edge, got {rag_non_adjacent.ecount()}"
    edge_labels = set()
    for edge in rag_non_adjacent.get_edgelist():
        edge_labels.add(tuple(sorted((rag_non_adjacent.vs[edge[0]]["name"], rag_non_adjacent.vs[edge[1]]["name"]))))
    assert edge_labels == {(1, 2)}, f"Expected edges {(1, 2)}, got {edge_labels}"
    print("Test Case 3 Passed: Non-Adjacent Same Labels")
    
    # Test Case 4: Complex Segmentation
    print("Running Test Case 4: Complex Segmentation")
    seg_complex = np.array([
        [1, 1, 2, 2],
        [1, 3, 3, 2],
        [4, 3, 3, 5],
        [4, 4, 5, 5]
    ])
    rag_complex = build_rag(seg_complex)
    assert rag_complex.vcount() == 5, f"Expected 5 vertices, got {rag_complex.vcount()}"
    assert rag_complex.ecount() == 8, f"Expected 8 edges, got {rag_complex.ecount()}"
    expected_edges = {(1, 2), (1, 3), (1, 4), (2, 3), (2, 5), (3, 4), (3, 5), (4, 5)}
    edge_labels = set()
    for edge in rag_complex.get_edgelist():
        edge_labels.add(tuple(sorted((rag_complex.vs[edge[0]]["name"], rag_complex.vs[edge[1]]["name"]))))
    assert edge_labels == expected_edges, f"Expected edges {expected_edges}, got {edge_labels}"
    print("Test Case 4 Passed: Complex Segmentation")
    
    # Test Case 5: Empty Segmentation
    print("Running Test Case 5: Empty Segmentation")
    seg_empty = np.array([[]], dtype=int)
    rag_empty = build_rag(seg_empty)
    assert rag_empty.vcount() == 0, f"Expected 0 vertices, got {rag_empty.vcount()}"
    assert rag_empty.ecount() == 0, f"Expected 0 edges, got {rag_empty.ecount()}"
    print("Test Case 5 Passed: Empty Segmentation")
    
    print("All build_rag tests passed successfully.")

if __name__ == "__main__":
    print("Starting tests for build_rag...")
    test_build_rag()
    print("All tests completed successfully!")