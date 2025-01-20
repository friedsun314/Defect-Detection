import cv2 as cv
import numpy as np

# Load the image
blank = cv.imread('defective_examples/case1_inspected_image.tif', cv.IMREAD_UNCHANGED)

# Check if the image is loaded successfully
if blank is None:
    print("Error: Image not loaded. Check the file path or format.")
    exit()

# Convert grayscale to BGR if necessary
if len(blank.shape) == 2:  # Grayscale image
    blank = cv.cvtColor(blank, cv.COLOR_GRAY2BGR)

# Modify a region of the image
blank[149,334:340] = [0, 255, 0]  # Assign green color
blank[82:90,245:250] = [0, 255, 0]  # Assign green color
blank[97,82:90] = [0, 255, 0]  # Assign green color

# Display the image
cv.imshow('Defect', blank)
cv.waitKey(0)
cv.destroyAllWindows()