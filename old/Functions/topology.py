import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from gudhi import CubicalComplex
from scipy.ndimage import label
from skimage.feature import canny
from skimage.measure import regionprops

# Define the file path
image_path = "defective_examples/case1_reference_image.tif"

# Check if the file exists
if not os.path.exists(image_path):
    print(f"Error: The file '{image_path}' does not exist. Please check the path.")
else:
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: Unable to load the image. Please ensure the file is a valid image.")
    else:
        # Preprocessing
        normalized_image = image / 255.0  # Normalize to range [0, 1]
        binary_image = (normalized_image > 0.5).astype(np.uint8)  # Binarize with a threshold

        # Visualization: Original and Binary Images
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(image, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Binary Image (Threshold=0.5)")
        plt.imshow(binary_image, cmap='gray')
        plt.axis('off')
        plt.show()

        # Persistent Homology
        def persistent_homology(binary_image):
            cubical_complex = CubicalComplex(top_dimensional_cells=binary_image)
            persistence = cubical_complex.compute_persistence()

            # Extract barcodes and persistence diagram
            barcodes = cubical_complex.persistence_intervals_in_dimension(0)

            # Plot Persistence Barcode
            plt.figure(figsize=(10, 5))
            for i, (birth, death) in enumerate(barcodes):
                plt.plot([birth, death], [i, i], "b-", label="H0" if i == 0 else "")
            plt.title("Persistence Barcode")
            plt.xlabel("Filtration Value")
            plt.ylabel("Barcode Index")
            plt.legend()
            plt.show()

            # Plot Persistence Diagram
            plt.figure(figsize=(8, 8))
            plt.scatter(
                [b for b, d in barcodes],
                [d for b, d in barcodes],
                label="H0",
                color="blue",
                alpha=0.7,
            )
            plt.title("Persistence Diagram")
            plt.xlabel("Birth")
            plt.ylabel("Death")
            plt.axline((0, 0), slope=1, color="red", linestyle="--", label="y=x (Diagonal)")
            plt.legend()
            plt.show()

        print("Generating persistent homology visualizations...")
        persistent_homology(binary_image)

        # Noise Filtering with Varying Parameters
        def noise_filtering(binary_image, min_area):
            labeled_array, num_features = label(binary_image)
            filtered_image = np.zeros_like(binary_image)

            for region in regionprops(labeled_array):
                if region.area > min_area:  # Filter by minimum area
                    coords = region.coords
                    for coord in coords:
                        filtered_image[coord[0], coord[1]] = 1

            plt.figure(figsize=(8, 8))
            plt.title(f"Noise Filtered Image (Min Area={min_area})")
            plt.imshow(filtered_image, cmap='gray')
            plt.axis('off')
            plt.show()

        print("Applying noise filtering with different minimum areas...")
        for min_area in [10, 50, 100]:  # Experiment with different thresholds
            noise_filtering(binary_image, min_area)

        # Edge Detection for Shape Analysis
        def shape_analysis(image, sigma):
            edges = canny(image / 255.0, sigma=sigma)
            plt.figure(figsize=(8, 8))
            plt.title(f"Edge Detection (Sigma={sigma})")
            plt.imshow(edges, cmap='gray')
            plt.axis('off')
            plt.show()

        print("Performing shape analysis with different sigma values...")
        for sigma in [1.0, 2.0, 3.0]:  # Experiment with different sigmas
            shape_analysis(image, sigma)

        # Texture Analysis with Laplacian
        def texture_analysis(image, kernel_size):
            texture_image = cv2.Laplacian(image, cv2.CV_64F, ksize=kernel_size)
            plt.figure(figsize=(8, 8))
            plt.title(f"Texture Analysis (Laplacian, Kernel Size={kernel_size})")
            plt.imshow(np.abs(texture_image), cmap='gray')
            plt.axis('off')
            plt.show()

        print("Performing texture analysis with different kernel sizes...")
        for kernel_size in [1, 3, 5]:  # Experiment with different kernel sizes
            texture_analysis(image, kernel_size)

        # Object Segmentation
        def object_segmentation(binary_image):
            labeled_array, num_features = label(binary_image)
            segmented_image = np.zeros_like(image)
            for i in range(1, num_features + 1):
                segmented_image[labeled_array == i] = i * (255 // num_features)  # Assign unique values

            plt.figure(figsize=(8, 8))
            plt.title("Object Segmentation")
            plt.imshow(segmented_image, cmap='nipy_spectral')
            plt.axis('off')
            plt.show()

        print("Performing object segmentation...")
        object_segmentation(binary_image)