import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the reference image
reference_image_path = "defective_examples/case1_reference_image.tif"
reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)

# Ensure the image is loaded
if reference_image is None:
    raise FileNotFoundError("Reference image not found.")

# Step 1: Morphological Cleaning
# Define a structuring element
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Apply morphological opening (erosion followed by dilation)
opened_image = cv2.morphologyEx(reference_image, cv2.MORPH_OPEN, kernel)

# Apply morphological closing (dilation followed by erosion)
cleaned_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel)

# Step 2: Compute the Fourier Transform
# Compute the 2D Fourier Transform
f_transform = np.fft.fft2(cleaned_image)

# Shift the zero frequency component to the center
f_shift = np.fft.fftshift(f_transform)

# Compute the magnitude spectrum (log scale for better visualization)
magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)

# Step 3: Identify Dominant Frequencies
# Threshold for dominant frequency selection
dominance_threshold = np.percentile(magnitude_spectrum, 98)
dominant_frequencies = (magnitude_spectrum > dominance_threshold)

# Step 4: Visualize the Results
plt.figure(figsize=(15, 8))

# Original reference image
plt.subplot(2, 3, 1)
plt.imshow(reference_image, cmap="gray")
plt.title("Original Reference Image")
plt.axis("off")

# After morphological opening
plt.subplot(2, 3, 2)
plt.imshow(opened_image, cmap="gray")
plt.title("After Opening")
plt.axis("off")

# After morphological cleaning (opening + closing)
plt.subplot(2, 3, 3)
plt.imshow(cleaned_image, cmap="gray")
plt.title("After Opening + Closing")
plt.axis("off")

# Magnitude spectrum
plt.subplot(2, 3, 4)
plt.imshow(magnitude_spectrum, cmap="gray")
plt.title("FFT Magnitude Spectrum")
plt.axis("off")

# Dominant frequencies
plt.subplot(2, 3, 5)
plt.imshow(dominant_frequencies, cmap="gray")
plt.title("Dominant Frequencies (Thresholded)")
plt.axis("off")

plt.tight_layout()
plt.show()

# Save Results (Optional)
cv2.imwrite("cleaned_image.tif", cleaned_image)
cv2.imwrite("fft_magnitude_spectrum_cleaned.tif", (magnitude_spectrum / magnitude_spectrum.max() * 255).astype(np.uint8))
cv2.imwrite("dominant_frequencies_cleaned.tif", (dominant_frequencies * 255).astype(np.uint8))
print("Preprocessing with opening and closing, and FFT visualization completed.")