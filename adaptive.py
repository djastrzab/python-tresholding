import cv2 as cv
import numpy as np


def modified_gaussian(img_path, min_threshold, max_threshold, reach):
    original_img = cv.imread(img_path)
    grayscale_img = cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)
    blurred_grayscale = cv.GaussianBlur(grayscale_img, (reach, reach), 0)

    def threshold2(h, w):
        return blurred_grayscale[h][w]*(max_threshold/255)+min_threshold

    output = grayscale_img.copy()

    height, width = grayscale_img.shape
    for h in range(height):
        for w in range(width):
            if output[h][w] >= threshold2(h,w):
                output[h][w] = 255
            else:
                output[h][w] = 0

    return output


def gaussian(img_path, C, reach):
    original_img = cv.imread(img_path)
    grayscale_img = cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)
    output_img = cv.adaptiveThreshold(grayscale_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, reach, C)
    return output_img


if __name__ == '__main__':
    pass


