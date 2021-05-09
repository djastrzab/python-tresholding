import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt



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


def parabola_gaussian(img_path, min_threshold, max_threshold, reach):
    original_img = cv.imread(img_path)
    grayscale_img = cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)
    blurred_grayscale = cv.GaussianBlur(grayscale_img, (reach, reach), 0)

    x = [0, 127, 255]
    y = [min_threshold, 127, max_threshold]

    A = np.matrix([[el**i for i in range(3)] for el in x], dtype=np.float64)
    B = np.matrix(y, dtype=np.float64).T
    coef = np.linalg.solve(A, B).A1

    def plot():
        f = lambda x: coef[0]+x*coef[1]+(x**2)*coef[2]
        x = np.arange(0, 255, 0.1)
        y = [f(el) for el in x]

        fig, ax = plt.subplots()
        ax.plot(x,x)
        ax.plot(x,y)
        fig.savefig("thres")

    def threshold(h, w):
        x = blurred_grayscale[h][w]
        return coef[0]+x*coef[1]+(x**2)*coef[2]

    output = grayscale_img.copy()

    height, width = grayscale_img.shape
    for h in range(height):
        for w in range(width):
            if output[h][w] >= threshold(h,w):
                output[h][w] = 255
            else:
                output[h][w] = 0

    return output


def parabola_gaussian_improved(img_path, min_threshold, max_threshold, reach):
    original_img = cv.imread(img_path)
    grayscale_img = cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)
    blurred_grayscale = cv.GaussianBlur(grayscale_img, (reach, reach), 0)

    x = [0, 127, 255]
    y = [min_threshold, 127, max_threshold]

    A = np.matrix([[el**i for i in range(3)] for el in x], dtype=np.float64)
    B = np.matrix(y, dtype=np.float64).T
    coef = np.linalg.solve(A, B).A1

    def threshold(x):
        return coef[0]+x*coef[1]+(x**2)*coef[2]

    return (grayscale_img >= threshold(np.array(blurred_grayscale, dtype=np.float64)))*255


def gaussian(img_path, C, reach):
    original_img = cv.imread(img_path)
    grayscale_img = cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)
    output_img = cv.adaptiveThreshold(grayscale_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, reach, C)
    return output_img




