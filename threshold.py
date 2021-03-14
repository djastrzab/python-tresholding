import cv2 as cv
import numpy as np
import sys


def otsu(histogram):
    sum_of_pixels = sum(histogram)
    min_var = -np.inf
    threshold = 0

    for t in range(0, 256):
        prob_0 = np.sum(histogram[0:t]) / sum_of_pixels
        prob_1 = 1 - prob_0

        if prob_0 == 0 or prob_1 == 0:
            continue

        mean_0 = np.dot([i for i in range(0, t)], histogram[0:t]) / prob_0
        mean_1 = np.dot([i for i in range(t, 256)], histogram[t:256]) / prob_1

        var = prob_0 * prob_1 * (mean_0 - mean_1) * (mean_0 - mean_1)
        if var > min_var:
            min_var = var
            threshold = t
    return threshold


def process_figure_with_otsu(img_path):

    original_img = cv.imread(img_path)
    grayscale_img = cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)
    blurred_grayscale = cv.GaussianBlur(grayscale_img, (11, 11), 0)

    dark_filter = blurred_grayscale < 100
    light_filter = blurred_grayscale > 150
    medium_filer = np.vectorize(lambda x: 150 >= x >= 100)(blurred_grayscale)

    light_histogram = np.histogram(grayscale_img[light_filter], bins=256, range=(0, 256))
    light_threshold = otsu(light_histogram[0])

    dark_histogram = np.histogram(grayscale_img[dark_filter], bins=256, range=(0, 256))
    dark_threshold = otsu(dark_histogram[0])

    medium_histogram = np.histogram(grayscale_img[medium_filer], bins=256, range=(0, 256))
    medium_threshold = otsu(medium_histogram[0])

    print(dark_threshold,medium_threshold,light_threshold)

    output = grayscale_img.copy()

    height, width = grayscale_img.shape
    for h in range(height):
        for w in range(width):
            if dark_filter[h][w]:
                if output[h][w] > dark_threshold:
                    output[h][w] = 255
                else:
                    output[h][w] = 0
            elif light_filter[h][w]:
                if output[h][w] > light_threshold:
                    output[h][w] = 255
                else:
                    output[h][w] = 0
            else:
                if output[h][w] > medium_threshold:
                    output[h][w] = 255
                else:
                    output[h][w] = 0
    return output


def process_figure_with_gaussian(img_path):
    original_img = cv.imread(img_path)
    grayscale_img = cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)
    output_img = cv.adaptiveThreshold(grayscale_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 51, -10)
    cv.imwrite("outputGauss.tif", output_img)


def process_figure(img_path, lower_division_point, higher_division_point, dark_threshold, medium_threshold, light_threshold):
    original_img = cv.imread(img_path)
    grayscale_img = cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)
    blurred_grayscale = cv.GaussianBlur(grayscale_img, (11, 11), 0)

    dark_filter = blurred_grayscale < lower_division_point
    light_filter = blurred_grayscale > higher_division_point

    output = grayscale_img.copy()

    height, width = grayscale_img.shape
    for h in range(height):
        for w in range(width):
            if dark_filter[h][w]:
                if output[h][w] > dark_threshold:
                    output[h][w] = 255
                else:
                    output[h][w] = 0
            elif light_filter[h][w]:
                if output[h][w] > light_threshold:
                    output[h][w] = 255
                else:
                    output[h][w] = 0
            else:
                if output[h][w] > medium_threshold:
                    output[h][w] = 255
                else:
                    output[h][w] = 0
    return output


def improvised_gaussian_adaptive(img_path):
    original_img = cv.imread(img_path)
    grayscale_img = cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)
    blurred_grayscale = cv.GaussianBlur(grayscale_img, (51, 51), 0)

    def threshold(h, w):
        return blurred_grayscale[h][w] + 10

    output = grayscale_img.copy()

    height, width = grayscale_img.shape
    for h in range(height):
        for w in range(width):
            if output[h][w] > threshold(h,w):
                output[h][w] = 255
            else:
                output[h][w] = 0

    cv.imwrite("imprGauss.tif", output)


if __name__ == "__main__":
    improvised_gaussian_adaptive("original.jpg")


