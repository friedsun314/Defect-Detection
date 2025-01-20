import cv2 as cv
 
img = cv.imread('defective_examples/case1_inspected_image.tif', cv.IMREAD_UNCHANGED)

def rescaleFrame(frame, scale=1.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
 
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

resized_image = rescaleFrame(img)
cv.imshow('case1_inspected', resized_image)

cv.waitKey(0) 