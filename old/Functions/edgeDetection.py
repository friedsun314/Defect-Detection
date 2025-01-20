import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the reference image
reference_image_path = "defective_examples/case1_reference_image.tif"
reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)

# Ensure the image is loaded
if reference_image is None:
    raise FileNotFoundError("Reference image not found.")

# Step 1: Fourier Transform on Reference Image
# Compute the 2D Fourier Transform and shift the zero frequency component to the center
f_transform = np.fft.fft2(reference_image)
f_shift = np.fft.fftshift(f_transform)

# Compute the magnitude spectrum
magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)

# Step 2: Identify Dominant Frequencies
# Threshold for dominant frequency selection (adjust as needed)
dominance_threshold = np.percentile(magnitude_spectrum, 98)
dominant_frequencies = (magnitude_spectrum > dominance_threshold)

# Step 3: Create Value Discretization Map
rows, cols = reference_image.shape
labeled_image = np.zeros((rows, cols), dtype=np.uint8)
block_size = 17  # Block size for local frequency analysis

# Iterate over blocks
for row in range(0, rows, block_size):
    for col in range(0, cols, block_size):
        # Extract the block
        block = reference_image[row:row + block_size, col:col + block_size]
        
        # Perform Fourier Transform on the block
        f_block = np.fft.fft2(block)
        f_block_shift = np.fft.fftshift(f_block)
        magnitude_block = 20 * np.log(np.abs(f_block_shift) + 1)

        # Compare block frequencies to dominant frequencies
        match = np.sum((magnitude_block > dominance_threshold).astype(np.uint8))

        # Assign a value based on the match (arbitrary scaling for visualization)
        if match > 0:
            labeled_image[row:row + block_size, col:col + block_size] = match

# Step 4: Visualize Results
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(reference_image, cmap="gray")
plt.title("Original Reference Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(magnitude_spectrum, cmap="gray")
plt.title("Magnitude Spectrum")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(labeled_image, cmap="tab20")
plt.title("Discretized Value Map")
plt.axis("off")

plt.tight_layout()
plt.show()

# Save Results (Optional)
# cv2.imwrite("labeled_image_discretized.tif", labeled_image)
# print("Discretization completed and saved.")