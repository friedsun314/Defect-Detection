import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def divide_image_into_sets(image_path, n_clusters=4):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    
    # Reshape the image to a 2D array of pixels and their RGB values
    pixels = image.reshape(-1, 3)
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_.astype(int)
    
    # Reconstruct the segmented image
    segmented_image = centers[labels].reshape(image.shape)
    
    # Create masks for each cluster
    masks = []
    for i in range(n_clusters):
        mask = (labels == i).reshape(image.shape[:2])
        masks.append(mask)
    
    # Display the results
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(2, 3, 2)
    plt.imshow(segmented_image)
    plt.title("Segmented Image")
    plt.axis("off")
    
    # Display masks
    for i, mask in enumerate(masks):
        plt.subplot(2, 3, i + 3)
        plt.imshow(mask, cmap='gray')
        plt.title(f"Cluster {i + 1}")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()
    
    return masks, segmented_image

# Example usage
image_path = 'output/partitions/inspected_partition_1.png'  # Replace with your image path
divide_image_into_sets(image_path)