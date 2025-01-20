import cv2
import numpy as np
import matplotlib.pyplot as plt

def perform_opening_closing(image_path, iterations):
    # Read the input image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Unable to read the image. Please check the file path.")
        return
    
    # Create a structuring element (kernel)
    kernel = np.ones((3, 3), np.uint8)  # 3x3 kernel for opening and closing
    
    # Perform opening and closing operations iteratively
    opening_images = [image]  # Store original image for opening
    closing_images = [image]  # Store original image for closing
    
    opening_image = image.copy()
    closing_image = image.copy()
    
    for i in range(1, iterations + 1):
        # Opening = Erosion followed by Dilation
        opening_image = cv2.dilate(cv2.erode(opening_image, kernel, iterations=1), kernel, iterations=1)
        opening_images.append(opening_image)
        
        # Closing = Dilation followed by Erosion
        closing_image = cv2.erode(cv2.dilate(closing_image, kernel, iterations=1), kernel, iterations=1)
        closing_images.append(closing_image)
    
    # Plot the results
    plt.figure(figsize=(15, 10))
    
    # Plot opening results
    for i, img in enumerate(opening_images):
        plt.subplot(2, iterations + 1, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Opening {i}")
        plt.axis('off')
    
    # Plot closing results
    for i, img in enumerate(closing_images):
        plt.subplot(2, iterations + 1, iterations + 2 + i)
        plt.imshow(img, cmap='gray')
        plt.title(f"Closing {i}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Input from the user
image_path = input("Enter the path to the image file: ")
iterations = int(input("Enter the number of iterations: "))

# Call the function
perform_opening_closing(image_path, iterations)