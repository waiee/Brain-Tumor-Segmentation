import os
import cv2
import numpy as np
from skimage.measure import label
from skimage.morphology import remove_small_objects, footprints

# Function to get the largest connected component
def getLargestCC(segmentation):
    labels = label(segmentation)
    assert(labels.max() != 0)
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC.astype('uint8') * 255

# Input and output folder paths
input_folder = "Dataset/input/images"
output_folder = "Dataset/output/images"

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each image in the input folder
for filename in os.listdir(input_folder):
    # Load the image
    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path, 0)  # Read the image in grayscale

        # Apply median blur
    blur_median = cv2.medianBlur(img,1)

    #Sharpening
    blurred = cv2.GaussianBlur(img, (17, 17), 0)
    imgSharp = np.float32(img)
    imgDetails = imgSharp - blurred
    imgResult = imgSharp + imgDetails

    # convert back to 8bits gray scale
    imgResult = np.clip(imgResult, 0, 255)
    imgResult = imgResult.astype('uint8')
    blurred = np.clip(blurred, 0, 255)
    blurred = np.uint8(blurred)

    #Get the highest pixel intensity - presumably the skull
    high_pixel = 0.01
    hist = cv2.calcHist([imgResult],[0],None,[256],[0,256]).flatten()
    total_count = imgResult.shape[0] * imgResult.shape[1]  # height * width
    target_count = high_pixel * total_count # bright pixels we look for
    summed = 0
    for i in range(255, 0, -1):
        summed += int(hist[i])
        if target_count <= summed:
            hi_thresh = i
            break
        else:
            hi_thresh = 0
    filtered = cv2.threshold(imgResult,hi_thresh,0,cv2.THRESH_TOZERO)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
    img_close = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)
    img_dilate = cv2.dilate(img_close, kernel)


    # Thresholding
    thres, img_thres = cv2.threshold(blur_median, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #Subtract the threshold (otsu with the skull)
    strip = img_thres-filtered
    
    kernel = footprints.octagon(6,6)
    img_morph = cv2.morphologyEx(strip, cv2.MORPH_OPEN, kernel)

    img_cc = getLargestCC(img_morph)
    img_cc = img_cc.astype('uint8') * 255

    kernel = footprints.octagon(15,15)
    img_cc = cv2.morphologyEx(img_cc, cv2.MORPH_CLOSE, kernel)

    img_strip = cv2.bitwise_and(img, img, mask = img_cc)

    # Save the processed image to the output folder
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, img_strip)
