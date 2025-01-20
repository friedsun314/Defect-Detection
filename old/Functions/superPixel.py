import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "defective_examples/case1_inspected_image.tif"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError("The image path is incorrect.")

# Convert the image to LAB color space (required by SLIC in OpenCV)
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# SLIC Superpixel Segmentation
num_superpixels = 200  # Number of superpixels to generate
compactness = 10  # Balances color proximity and space proximity
slic = cv2.ximgproc.createSuperpixelSLIC(lab_image, region_size=20, ruler=compactness)
slic.iterate(10)  # Number of iterations for refinement

# Get the superpixel mask
mask_slic = slic.getLabelContourMask()  # Mask with superpixel boundaries
contours = np.where(mask_slic == 255)  # Contour pixels

# Overlay the superpixel boundaries on the original image
superpixel_image = image.copy()
superpixel_image[contours] = [0, 255, 0]  # Green boundaries

# Display the original image and superpixel segmentation
plt.figure(figsize=(12, 6))

# Original image
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")

# Superpixel segmentation
plt.subplot(1, 2, 2)
plt.title("Superpixel Segmentation")
plt.imshow(cv2.cvtColor(superpixel_image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()