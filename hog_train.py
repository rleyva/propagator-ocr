__author__ = 'Ralph F. Leyva'

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

class dataset():
    def __init__(self):
        self.samples = np.genfromtxt('final_samples.data').astype(np.float32)
        self.responses = np.genfromtxt('final_responses.data').astype(np.float32)

digit_repo = dataset()
digits = np.array(digit_repo.samples, 'float32')
labels = np.array(digit_repo.responses, 'float32')

list_hog_fd = []
list_hog_images = []
list_digit = []

#for digit in digits:
#    list_digit.append(digit.reshape(10, 10))

for digit in digits:
    fd, hog_image = hog(digit.reshape(10,10), orientations=2, pixels_per_cell=(2,2), cells_per_block=(1,1), visualise=True)
    list_hog_fd.append(fd)
    list_hog_images.append(hog_image)
hog_features = np.array(list_hog_fd, 'float32')

cv2.imshow('', list_hog_images[540])
cv2.waitKey(0)

clf = LinearSVC()
clf.fit(hog_features, labels)
joblib.dump(clf, "digits_cls.pkl", compress=3)