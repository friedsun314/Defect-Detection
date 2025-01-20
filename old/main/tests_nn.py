import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.api.models import Sequential
from keras.api.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.api.optimizers import Adam
import random
from main.merging import find_neighbors, prepare_dataset, fine_tune_model, enlarge_seeds_with_nn

# Step 1: Load a sample grayscale image
image_path = "defective_examples/case1_inspected_image.tif"  # Replace with a valid path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("Image not found. Please check the path.")

# Step 2: Test find_neighbors
print("Testing find_neighbors...")
sample_seed = [(10, 10), (11, 10), (10, 11)]  # Example seed
neighbors = find_neighbors(sample_seed, image.shape)
print("Sample Seed:", sample_seed)
print("Neighbors:", neighbors)

# Step 3: Test prepare_dataset
print("Testing prepare_dataset...")
seeds = [sample_seed]  # For testing, use a single seed
X, y = prepare_dataset(image, seeds, patch_size=5, n_random_samples=5)
print("Dataset prepared. X shape:", X.shape, "y shape:", y.shape)

# Visualize a few patches
plt.figure(figsize=(10, 5))
for i in range(min(5, len(X))):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X[i].squeeze(), cmap='gray')
    plt.title(f"Label: {y[i]}")
    plt.axis('off')
plt.show()

# Step 4: Test fine_tune_model
print("Testing fine_tune_model...")
model = fine_tune_model(input_shape=(5, 5, 1))
model.summary()

# Train the model briefly (for testing)
print("Training model...")
model.fit(X, y, epochs=1, batch_size=2)

# Step 5: Test enlarge_seeds_with_nn
print("Testing enlarge_seeds_with_nn...")
enlarged_seeds = enlarge_seeds_with_nn(image, seeds, model, patch_size=5, max_iterations=1)
print("Enlarged Seeds:", enlarged_seeds)

# Visualize the enlarged seeds on the image
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for seed in enlarged_seeds:
    for x, y in seed:
        cv2.circle(output_image, (x, y), radius=1, color=(0, 255, 0), thickness=-1)

plt.figure(figsize=(10, 10))
plt.imshow(output_image)
plt.title("Enlarged Seeds Visualization")
plt.axis('off')
plt.show()