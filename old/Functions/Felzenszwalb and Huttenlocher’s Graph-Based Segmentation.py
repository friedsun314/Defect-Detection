import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Union-Find (Disjoint Set) Helpers
# -----------------------------
def make_set(n):
    """Initialize parent, rank, and component size for n elements."""
    parent = np.arange(n)
    rank = np.zeros(n, dtype=np.int32)
    size = np.ones(n, dtype=np.int32)
    return parent, rank, size

def find_set(parent, x):
    """Path compression find."""
    if parent[x] != x:
        parent[x] = find_set(parent, parent[x])
    return parent[x]

def union_set(parent, rank, size, a, b):
    """Union by rank; update component sizes."""
    a_root = find_set(parent, a)
    b_root = find_set(parent, b)
    if a_root != b_root:
        if rank[a_root] < rank[b_root]:
            parent[a_root] = b_root
            size[b_root] += size[a_root]
        elif rank[a_root] > rank[b_root]:
            parent[b_root] = a_root
            size[a_root] += size[b_root]
        else:
            parent[b_root] = a_root
            rank[a_root] += 1
            size[a_root] += size[b_root]

# -----------------------------
# Felzenszwalb-Huttenlocher Segmentation
# -----------------------------
def felzenszwalb_segmentation(img, k=300, min_size=50):
    """
    Basic Felzenszwalb and Huttenlocher's segmentation.
    
    :param img: Input image (grayscale or RGB).
    :param k: Threshold constant for component merge criterion (higher => fewer merges).
    :param min_size: Minimum component size to enforce after merges.
    :return: label_map where each pixel has a component label.
    """
    # Ensure we have a 3D array (for both grayscale or color)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w, _ = img.shape
    
    # Number of pixels
    n = h * w
    
    # 1) Build edges (4-connected or 8-connected; here we do 4-connected for simplicity)
    edges = []
    def idx(r, c):
        return r * w + c
    
    for r in range(h):
        for c in range(w):
            if c + 1 < w:  # horizontal neighbor
                diff = color_diff(img[r, c], img[r, c+1])
                edges.append((idx(r, c), idx(r, c+1), diff))
            if r + 1 < h:  # vertical neighbor
                diff = color_diff(img[r, c], img[r+1, c])
                edges.append((idx(r, c), idx(r+1, c), diff))
    
    # 2) Sort edges by weight (color difference)
    edges.sort(key=lambda e: e[2])
    
    # 3) Initialize Union-Find structure
    parent, rank, size = make_set(n)
    
    # 4) Internal difference threshold: track max edge weight in each component
    #    (initially 0, updated upon merges)
    threshold_comp = np.zeros(n, dtype=np.float32)
    
    # 5) Kruskal-like merging
    for p1, p2, wght in edges:
        root1 = find_set(parent, p1)
        root2 = find_set(parent, p2)
        if root1 != root2:
            # If edge weight is small compared to the internal variation => merge
            if wght <= threshold_comp[root1] + k/size[root1] and \
               wght <= threshold_comp[root2] + k/size[root2]:
                union_set(parent, rank, size, root1, root2)
                new_root = find_set(parent, root1)
                threshold_comp[new_root] = wght
    
    # 6) Enforce minimum component size
    for p1, p2, wght in edges:
        root1 = find_set(parent, p1)
        root2 = find_set(parent, p2)
        if root1 != root2 and (size[root1] < min_size or size[root2] < min_size):
            union_set(parent, rank, size, root1, root2)
    
    # 7) Create label map
    #    Compress once more so each node points to its final root
    for i in range(n):
        parent[i] = find_set(parent, i)
    
    # Re-map root IDs to a new labeling [0..num_segments-1]
    unique_roots, labels = np.unique(parent, return_inverse=True)
    label_map = labels.reshape((h, w))
    
    return label_map

def color_diff(c1, c2):
    """Euclidean distance in RGB space."""
    return np.sqrt(np.sum((c1.astype(np.float32) - c2.astype(np.float32))**2))

# -----------------------------
# Visualization
# -----------------------------
def plot_segmentation(input_img, label_map, title="Felzenszwalb Segmentation"):
    """
    Visualize segmented output by coloring each label differently.
    """
    h, w = label_map.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)

    # Assign random color per segment
    np.random.seed(42)  # For reproducibility
    max_label = label_map.max()
    colors = [tuple(np.random.randint(0, 255, size=3)) for _ in range(max_label+1)]
    
    for r in range(h):
        for c in range(w):
            output[r, c] = colors[label_map[r, c]]
    
    # Show side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # Original
    axes[0].imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis("off")
    # Segmentation
    axes[1].imshow(output)
    axes[1].set_title(title)
    axes[1].axis("off")
    plt.show()

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    # Load an example image (color)
    img_path = "defective_examples/case1_inspected_image.tif"
    img_bgr = cv2.imread(img_path)
    
    # Run Felzenszwalb-Huttenlocher segmentation
    label_map = felzenszwalb_segmentation(img_bgr, k=300, min_size=50)
    
    # Visualize
    plot_segmentation(img_bgr, label_map, title="Felzenszwalb Segmentation")