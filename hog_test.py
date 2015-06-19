__author__ = 'Ralph F. Leyva'

# Module imports
import cv2
import json
import numpy as np
from StringIO import StringIO
from sklearn.externals import joblib
from skimage.feature import hog

'''
    PROPAGATOR OCR SERVER
        This script will implement a Linear SVC/HOG based OCR client
        to recognize images being passed through JSON which contain
        hexadecimal-like digits constructed out of pipes. Sample images
        can be found in the Digits folder.
'''

# HSV color-ranges used to threshold out the hexadecimal value
orange_min_hsv = np.array([0, 50, 50],np.uint8)
orange_max_hsv = np.array([10, 255, 255],np.uint8)

def convert_bgr2greyscale(cv_image):
    # Converts the image from BGR colorspace to greyscale
    return cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

def convert_bgr2hsv(cv_image):
    # Converts the image from BGR colorspace to HSV inorder to threshold
    return cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

def resize_image(image, scaling_factor):
    # Resizes the image...
    return cv2.resize(image, (0,0), fx = scaling_factor, fy = scaling_factor)

def transform_image(image):
    # Carries out the appropriate transforms on the given input image. Here
    # we carry out thresholding for orange, and attempt to blur each of the
    # segments for the image to make one continuous blob that we can
    # treat as a hex-value.
    resized_image = resize_image(image, 0.2)
    hsv_cv_image = convert_bgr2hsv(resized_image)
    orange_mask_hsv = cv2.inRange(hsv_cv_image, orange_min_hsv, orange_max_hsv)
    mask_blur = cv2.blur(orange_mask_hsv,(20,20))
    return cv2.threshold(mask_blur, 50, 255, cv2.THRESH_BINARY)[1]

def edge_detect(image):
    # As the name suggests, this function returns the edges currently in the image
    # along with the hierarchy. We only need the contours in this case since we will
    # not be interested in inner-contours of numbers.
    edges = cv2.Canny(image, 100, 255)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours, hierarchy

def find_max_contour(contours):
    # Returns the largest contour in the images (We're assuming that this is the
    # number that we're looking for)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    return contours[max_index]

def cast_char(value):
    if value == 10:
        return 'A'
    elif value == 11:
        return 'B'
    elif value == 12:
        return 'C'
    elif value == 13:
        return 'D'
    elif value == 14:
        return 'E'
    elif value == 15:
        return 'F'
    else:
        return str(value)

def recognize_char(image):
    # Calculates HOG features of the incoming image & returns the predicted character
    fd_roi_hog, hd_image = hog(roi_small_sqr, orientations=2, pixels_per_cell=(2, 2), cells_per_block=(1, 1), visualise=True)
    nbr = clf.predict(np.array([fd_roi_hog], 'float32'))
    return cast_char(nbr[0])

if __name__=="__main__":
    # Unpacks the pre-trained digit classifier
    clf = joblib.load("digits_cls.pkl")

    cv_image = cv2.imread('Digits/A.jpg')
    cv_image = transform_image(cv_image)
    contours, hierarchy = edge_detect(cv_image)
    max_contour = find_max_contour(contours)

    # Generate bounding rectangle which contains our region of interest
    [x, y, w, h] = cv2.boundingRect(max_contour)
    cv2.rectangle(cv_image, (x,y), (x+w, y+y+h), (0,255,0), 2)

    roi = cv_image[y:y+h, x:x+w]
    roi_small_sqr = cv2.resize(roi,(10,10))
    print recognize_char(roi_small_sqr)