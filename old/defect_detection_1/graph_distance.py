import igraph as ig
import numpy as np

def compute_graph_distance(rag_ref, rag_insp, seg_ref, seg_insp, intensity_ref, intensity_insp):
    """
    Compute a combined distance between two RAGs based on structural and intensity differences.
    """
    # Input validation
    if seg_ref.shape != seg_insp.shape:
        raise ValueError("Segmented reference and inspected frames must have the same dimensions.")
    if intensity_ref.shape != intensity_insp.shape:
        raise ValueError("Intensity reference and inspected frames must have the same dimensions.")
    if seg_ref.shape != intensity_ref.shape or seg_insp.shape != intensity_insp.shape:
        raise ValueError("Segmentation and intensity frames must match in dimensions.")
    if not rag_ref.vcount() or not rag_insp.vcount():
        return 0.0  # No vertices in either graph

    # Structural distance using Jaccard similarity
    def get_edge_set(rag):
        if "name" not in rag.vs.attributes():
            raise ValueError("Vertices must have 'name' attributes.")
        return set(
            tuple(sorted((rag.vs[edge[0]]["name"], rag.vs[edge[1]]["name"])))
            for edge in rag.get_edgelist()
        )

    edges_ref = get_edge_set(rag_ref)
    edges_insp = get_edge_set(rag_insp)

    intersection = edges_ref.intersection(edges_insp)
    union = edges_ref.union(edges_insp)

    structural_distance = 1.0 - (len(intersection) / len(union)) if union else 0.0

    # Intensity distance: Compute average intensity per region
    def average_intensity(seg, intensity):
        sums = np.bincount(seg.ravel(), weights=intensity.ravel())
        counts = np.bincount(seg.ravel())
        avg_intensity = np.divide(sums, counts, out=np.zeros_like(sums, dtype=float), where=counts > 0)
        return avg_intensity

    avg_ref = average_intensity(seg_ref, intensity_ref)
    avg_insp = average_intensity(seg_insp, intensity_insp)

    # Compute intensity differences between common regions
    common_labels = set(rag_ref.vs["name"]).intersection(rag_insp.vs["name"])
    intensity_distance = (
        np.mean([abs(avg_ref[label] - avg_insp[label]) for label in common_labels]) / 255.0
        if common_labels
        else 0.0
    )

    # Combine distances
    combined_distance = 0.5 * structural_distance + 0.5 * intensity_distance
    return combined_distance
    
def build_simple_rag(segmentation):
    """
    Build a simple RAG for testing purposes using igraph.
    """
    # Validate segmentation labels are integers
    if not np.issubdtype(segmentation.dtype, np.integer):
        raise ValueError("Segmentation labels must be integers.")
    
    labels = np.unique(segmentation)
    rag = ig.Graph(directed=False)
    rag.add_vertices(labels.tolist())
    label_to_id = {label: idx for idx, label in enumerate(labels)}

    h, w = segmentation.shape
    edges = set()
    for y in range(h):
        for x in range(w):
            current_label = segmentation[y, x]
            if x < w - 1:  # Right neighbor
                neighbor_label = segmentation[y, x + 1]
                if neighbor_label != current_label:
                    edges.add((label_to_id[current_label], label_to_id[neighbor_label]))
            if y < h - 1:  # Bottom neighbor
                neighbor_label = segmentation[y + 1, x]
                if neighbor_label != current_label:
                    edges.add((label_to_id[current_label], label_to_id[neighbor_label]))
    rag.add_edges(edges)
    return rag

#############################
# Tests
#############################

def test_compute_graph_distance():
    print("Running test_compute_graph_distance...")
    seg_ref = np.array([
        [1, 1, 1],
        [1, 2, 1],
        [1, 1, 1]
    ])
    seg_insp = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ])
    intensity_ref = np.array([
        [10, 10, 10],
        [10, 200, 10],
        [10, 10, 10]
    ], dtype=np.uint8)
    intensity_insp = np.array([
        [10, 10, 10],
        [10, 10, 10],
        [10, 10, 10]
    ], dtype=np.uint8)
    rag_ref = build_simple_rag(seg_ref)
    rag_insp = build_simple_rag(seg_insp)
    distance = compute_graph_distance(rag_ref, rag_insp, seg_ref, seg_insp, intensity_ref, intensity_insp)
    assert np.isclose(distance, 0.5), f"Test failed: {distance}"
    print("test_compute_graph_distance passed.")

def test_identical_graphs():
    print("Running test_identical_graphs...")
    seg = np.array([
        [1, 1],
        [1, 1]
    ])
    intensity = np.array([
        [50, 50],
        [50, 50]
    ], dtype=np.uint8)
    rag_ref = build_simple_rag(seg)
    rag_insp = build_simple_rag(seg)
    distance = compute_graph_distance(rag_ref, rag_insp, seg, seg, intensity, intensity)
    assert np.isclose(distance, 0.0), f"Test failed: {distance}"
    print("test_identical_graphs passed.")

def test_empty_graphs():
    print("Running test_empty_graphs...")
    seg = np.array([[]], dtype=int)  # Empty 2D array with integer type
    intensity = np.array([[]], dtype=np.uint8)  # Empty intensity array
    rag_ref = build_simple_rag(seg)
    rag_insp = build_simple_rag(seg)
    distance = compute_graph_distance(rag_ref, rag_insp, seg, seg, intensity, intensity)
    assert np.isclose(distance, 0.0), f"Test failed: {distance}"
    print("test_empty_graphs passed.")
    
def test_mismatched_sizes():
    print("Running test_mismatched_sizes...")
    seg_ref = np.array([
        [1, 1],
        [1, 2]
    ])
    seg_insp = np.array([
        [1, 1, 1],
        [1, 2, 2]
    ])
    intensity_ref = np.array([
        [100, 100],
        [100, 200]
    ], dtype=np.uint8)
    intensity_insp = np.array([
        [100, 100, 100],
        [100, 200, 200]
    ], dtype=np.uint8)
    try:
        compute_graph_distance(
            build_simple_rag(seg_ref),
            build_simple_rag(seg_insp),
            seg_ref, seg_insp,
            intensity_ref, intensity_insp
        )
    except ValueError as e:
        print("test_mismatched_sizes passed.")
    else:
        assert False, "Test failed: Exception not raised for mismatched sizes."

def test_non_integer_labels():
    print("Running test_non_integer_labels...")
    seg = np.array([
        [1.1, 1.1],
        [1.1, 2.2]
    ])
    intensity = np.array([
        [100, 100],
        [100, 200]
    ], dtype=np.uint8)
    try:
        build_simple_rag(seg)
    except ValueError:
        print("test_non_integer_labels passed.")
    else:
        assert False, "Test failed: Exception not raised for non-integer labels."

if __name__ == "__main__":
    print("Starting tests...")
    test_compute_graph_distance()
    test_identical_graphs()
    test_empty_graphs()
    test_mismatched_sizes()
    test_non_integer_labels()
    print("All tests completed successfully!")