

import numpy as np
import cv2
from matplotlib import pyplot as plt



def threshold():
    img = cv2.imread('orginal.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(img_gray, (51, 51), 0)

    img_gray_hist = cv2.calcHist([blur], [0], None, [256], [0, 256])
    th3 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)

    plt.hist(blur.ravel(), 256, [0, 256]);
    plt.show()

    cv2.imwrite('test.tif', th3)
    cv2.waitKey()

if __name__ == '__main__':
    threshold()

