import cv2
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from collections import defaultdict
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

def extract_seed_features(image, seeds):
    """
    Extract features for each seed based on pixel intensity values.

    Args:
        image (np.ndarray): Grayscale input image.
        seeds (list): List of seed pixel groups, where each group is a list of (x, y) coordinates.

    Returns:
        np.ndarray: Feature matrix (rows = seeds, columns = features).
    """
    features = []
    for seed in seeds:
        # Ensure seed is a list of (x, y) coordinates
        intensities = [image[y, x] for x, y in seed]
        # Compute features (mean intensity, variance)
        mean_intensity = np.mean(intensities)
        intensity_variance = np.var(intensities)
        features.append([mean_intensity, intensity_variance])
    return np.array(features)

def cluster_seeds_with_overlap(image, seeds, n_clusters=5):
    """
    Cluster seeds into types based on feature similarity using K-means.
    Detect overlapping pixels and merge types accordingly.

    Args:
        image (np.ndarray): Grayscale input image.
        seeds (list): List of seed pixel groups, where each group is a list of (x, y) coordinates.
        n_clusters (int): Initial number of clusters (types).

    Returns:
        np.ndarray: Image with seeds colored by merged cluster type.
        np.ndarray: Labels for each seed after merging.
    """
    # Extract features for clustering
    features = extract_seed_features(image, seeds)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    initial_labels = kmeans.fit_predict(features)

    # Build a graph to detect overlaps
    num_clusters = n_clusters
    overlap_graph = defaultdict(set)

    # Check for overlapping pixels between seed sets
    seed_to_labels = defaultdict(set)
    for seed, label in zip(seeds, initial_labels):
        for pixel in seed:
            seed_to_labels[pixel].add(label)

    # Create edges between types with overlapping pixels
    for pixel, labels in seed_to_labels.items():
        labels = list(labels)
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                overlap_graph[labels[i]].add(labels[j])
                overlap_graph[labels[j]].add(labels[i])

    # Convert the graph into an adjacency matrix
    adjacency_matrix = np.zeros((num_clusters, num_clusters), dtype=int)
    for node, neighbors in overlap_graph.items():
        for neighbor in neighbors:
            adjacency_matrix[node, neighbor] = 1
            adjacency_matrix[neighbor, node] = 1

    # Find connected components (merged types)
    n_components, merged_labels = connected_components(csgraph=csr_matrix(adjacency_matrix), directed=False)

    # Reassign merged labels
    label_mapping = {old_label: new_label for old_label, new_label in enumerate(merged_labels)}
    final_labels = [label_mapping[label] for label in initial_labels]

    # Assign colors to merged types
    colors = plt.cm.get_cmap("tab20", n_components)
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for seed, label in zip(seeds, final_labels):
        color = tuple(int(c * 255) for c in colors(label)[:3])  # Cluster color
        for x, y in seed:
            cv2.circle(output_image, (x, y), radius=1, color=color, thickness=-1)

    return output_image, final_labels