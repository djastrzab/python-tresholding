

import numpy as np
import cv2
from matplotlib import pyplot as plt



def threshold():
    img = cv2.imread('orginal.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.threshold(img, THRESH, MAXVALUE, cv2.METHOD)
    # ret,img_bin1
    # cv2.adaptiveThreshold(img, maxvalue, Adaptive Method, BlockSize, C)
    # th3 = cv2.adaptiveThreshold(img_gray,100,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    # cv2.imshow('bin',th3)
    # cv2.imwrite('out.tif', th3)
    # cv2.waitKey(5000)
    # equ = cv2.equalizeHist(img)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32,32))
    # cl1 = clahe.apply(img)
    # res2 = np.hstack((img,cl1))
    # cv2.imwrite('hist_clahe.tif',cl1)
    # th2=cv2.adaptiveThreshold(img_gray,160,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH)
    # th3 = cv2.adaptiveThreshold(img_gray, 160, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,23)

    # img = cv.imread('home.jpg', 0)

    blur = cv2.GaussianBlur(img_gray, (51, 51), 0)

    img_gray_hist = cv2.calcHist([blur], [0], None, [256], [0, 256])
    th3 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)

    plt.hist(blur.ravel(), 256, [0, 256]);
    plt.show()

    cv2.imwrite('test.tif', th3)
    cv2.waitKey()

if __name__ == '__main__':
    threshold()

