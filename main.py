# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import numpy as np
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm' )

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
img=cv2.imread('source.tif',0)


#img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.threshold(img, THRESH, MAXVALUE, cv2.METHOD)
#ret,img_bin1
#cv2.adaptiveThreshold(img, maxvalue, Adaptive Method, BlockSize, C)
#th3 = cv2.adaptiveThreshold(img_gray,100,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
#cv2.imshow('bin',th3)
#cv2.imwrite('out.tif', th3)
#cv2.waitKey(5000)
equ = cv2.equalizeHist(img)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32,32))
cl1 = clahe.apply(img)
#res2 = np.hstack((img,cl1))
cv2.imwrite('hist_clahe.tif',cl1)
a=[1,2]
b=a.copy()