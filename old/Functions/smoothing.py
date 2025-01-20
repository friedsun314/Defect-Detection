import cv2
import matplotlib.pyplot as plt

# Load the inspected image
inspected_image_path = "defective_examples/case1_inspected_image.tif"
inspected_image = cv2.imread(inspected_image_path, cv2.IMREAD_GRAYSCALE)

# Ensure the image is loaded
if inspected_image is None:
    raise FileNotFoundError("Inspected image not found.")

# Initialize variables
iterations = 5
smoothed_images = [inspected_image]

# Repeatedly apply GaussianBlur
for i in range(iterations):
    smoothed_image = cv2.GaussianBlur(smoothed_images[-1], (5, 5), 0)
    smoothed_images.append(smoothed_image)

# Visualize the Results
plt.figure(figsize=(15, 10))
for i, img in enumerate(smoothed_images):
    plt.subplot(2, 3, i + 1)
    plt.imshow(img, cmap="gray")
    if i == 0:
        plt.title(f"Original Image")
    else:
        plt.title(f"GaussianBlur - Iteration {i}")
    plt.axis("off")

plt.tight_layout()
plt.show()

# Save Results (Optional)
# for i, img in enumerate(smoothed_images):
#     filename = f"gaussian_smoothed_iteration_{i}.tif"
#     cv2.imwrite(filename, img)
# print("Repeated smoothing completed and results saved.")