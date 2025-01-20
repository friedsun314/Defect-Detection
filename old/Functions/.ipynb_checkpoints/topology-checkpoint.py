import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from gudhi import CubicalComplex
from scipy.ndimage import label
from skimage.feature import canny
from skimage.measure import regionprops
from ipywidgets import interact

# Define the file path
image_path = "defective_examples/case1_inspected_image.tif"

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

        # Function: Display the original and processed images
        def visualize_images(show_binary=False):
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("Original Image")
            plt.imshow(image, cmap='gray')
            plt.axis('off')

            if show_binary:
                plt.subplot(1, 2, 2)
                plt.title("Binary Image")
                plt.imshow(binary_image, cmap='gray')
                plt.axis('off')

            plt.show()

        # Function: Persistent Homology Visualization
        def persistent_homology():
            cubical_complex = CubicalComplex(top_dimensional_cells=binary_image)
            cubical_complex.compute_persistence()

            # Plot Persistence Barcode
            plt.figure(figsize=(10, 5))
            plt.title("Persistence Barcode")
            cubical_complex.plot_persistence_barcode()
            plt.show()

            # Plot Persistence Diagram
            plt.figure(figsize=(8, 8))
            plt.title("Persistence Diagram")
            cubical_complex.plot_persistence_diagram()
            plt.show()

        # Function: Noise Filtering
        def noise_filtering():
            labeled_array, num_features = label(binary_image)
            filtered_image = np.zeros_like(binary_image)

            # Filter out small components
            for region in regionprops(labeled_array):
                if region.area > 50:  # Keep regions with area > 50
                    coords = region.coords
                    for coord in coords:
                        filtered_image[coord[0], coord[1]] = 1

            plt.figure(figsize=(8, 8))
            plt.title("Filtered Image")
            plt.imshow(filtered_image, cmap='gray')
            plt.axis('off')
            plt.show()

        # Function: Edge Detection for Shape Analysis
        def shape_analysis():
            edges = canny(image / 255.0)
            plt.figure(figsize=(8, 8))
            plt.title("Edge Detection for Shape Analysis")
            plt.imshow(edges, cmap='gray')
            plt.axis('off')
            plt.show()

        # Function: Texture Analysis
        def texture_analysis():
            texture_image = cv2.Laplacian(image, cv2.CV_64F)
            plt.figure(figsize=(8, 8))
            plt.title("Texture Analysis (Laplacian)")
            plt.imshow(np.abs(texture_image), cmap='gray')
            plt.axis('off')
            plt.show()

        # Function: Object Segmentation
        def object_segmentation():
            labeled_array, num_features = label(binary_image)
            segmented_image = np.zeros_like(image)
            for i in range(1, num_features + 1):
                segmented_image[labeled_array == i] = i * 40  # Assign unique values

            plt.figure(figsize=(8, 8))
            plt.title("Object Segmentation")
            plt.imshow(segmented_image, cmap='nipy_spectral')
            plt.axis('off')
            plt.show()

        # Interactive Menu
        @interact(
            option=[
                "Visualize Images",
                "Persistent Homology",
                "Noise Filtering",
                "Shape Analysis",
                "Texture Analysis",
                "Object Segmentation",
            ]
        )
        def interactive_visualization(option):
            if option == "Visualize Images":
                visualize_images(show_binary=True)
            elif option == "Persistent Homology":
                persistent_homology()
            elif option == "Noise Filtering":
                noise_filtering()
            elif option == "Shape Analysis":
                shape_analysis()
            elif option == "Texture Analysis":
                texture_analysis()
            elif option == "Object Segmentation":
                object_segmentation()