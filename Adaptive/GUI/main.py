import imutils
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import cv2 as cv
import numpy as np
import sys
import matplotlib.pyplot as plt


def parabola_gaussian_img(img, blurred_img, min_threshold, mid_threshold,max_threshold, reach):

    x = [0, 127, 255]
    y = [min_threshold, mid_threshold, max_threshold]

    A = np.matrix([[el**i for i in range(3)] for el in x], dtype=np.float64)
    B = np.matrix(y, dtype=np.float64).T
    coef = np.linalg.solve(A, B).A1

    def threshold(x):
        return coef[0]+x*coef[1]+(x**2)*coef[2]

    return np.array((img >= threshold(np.array(blurred_img, dtype=np.float64)))*255, dtype=np.uint8)


def make_slider(v_from, v_to):
    slider = QSlider(Qt.Vertical)
    slider.setMaximum(v_to)
    slider.setMinimum(v_from)
    slider.setTickInterval(1)
    return slider


if __name__ == '__main__':
    app = QApplication([])
    window = QWidget()
    slidersAndPictureBox = QHBoxLayout()
    image_to_process = None
    blurred_img = None
    result_img = None


    def resize_img(image):
        screen_resolution = QApplication.primaryScreen()
        height = screen_resolution.size().height() - 140
        view_image = imutils.resize(image, height=height)
        return view_image

    def process_img():
        global result_img
        if image_to_process is not None:
            # print(minSlider.value(), midSlider.value(),maxSlider.value(), reachSlider.value())
            result_img = parabola_gaussian_img(image_to_process, blurred_img, minSlider.value(), midSlider.value(), maxSlider.value(), reachSlider.value())
            view_image = resize_img(result_img)
            h, w = view_image.shape
            pix = QPixmap.fromImage(QImage(view_image.data, w, h, w, QImage.Format.Format_Grayscale8))
            label_imageDisplay.setPixmap(pix.scaled(label_imageDisplay.size(), Qt.KeepAspectRatio))

    def open_image():
        global image_to_process
        global blurred_img
        reach = reachSlider.value()
        if reach%2 == 0:
            reach+=1
        filename = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
        if filename == "":
            return
        image_to_process = cv.imread(filename, cv.IMREAD_GRAYSCALE)
        blurred_img = cv.GaussianBlur(image_to_process, (reach, reach), 0)
        print(image_to_process.data)
        view_image = resize_img(image_to_process)
        h, w = view_image.shape
        pix = QPixmap.fromImage(QImage(view_image.data, w, h, w, QImage.Format.Format_Grayscale8))
        label_imageDisplay.setPixmap(pix)


    def save_image():
        if result_img is None:
            return
        filename = QFileDialog.getSaveFileName(filter="Image (*.tif)")[0]
        if filename == "":
            return
        print(filename)
        cv.imwrite(filename, result_img)

    minSliderBox = QVBoxLayout()
    minSlider = make_slider(0, 255)
    minSlider.setMinimum(-20)
    minSliderBox.addWidget(minSlider)
    minSliderLabel = QLabel(str(minSlider.value()))
    minSliderBox.addWidget(QLabel("min"))
    minSliderBox.addWidget(minSliderLabel)

    def min_value_changed(new_value):
        minSliderLabel.setText(str(new_value))
        process_img()
    minSlider.valueChanged.connect(min_value_changed)

    midSliderBox = QVBoxLayout()
    midSlider = make_slider(0, 255)
    midSliderBox.addWidget(midSlider)
    midSlider.setValue(127)
    midSliderLabel = QLabel(str(midSlider.value()))
    midSliderBox.addWidget(QLabel("mid"))
    midSliderBox.addWidget(midSliderLabel)

    def mid_value_changed(new_value):
        midSliderLabel.setText(str(new_value))
        process_img()
    midSlider.valueChanged.connect(mid_value_changed)

    maxSliderBox = QVBoxLayout()
    maxSlider = make_slider(0, 255)
    maxSliderBox.addWidget(maxSlider)
    maxSlider.setValue(255)
    maxSliderLabel = QLabel(str(maxSlider.value()))
    maxSliderBox.addWidget(QLabel("max"))
    maxSliderBox.addWidget(maxSliderLabel)

    def max_value_changed(new_value):
        maxSliderLabel.setText(str(new_value))
        process_img()
    maxSlider.valueChanged.connect(max_value_changed)

    reachSliderBox = QVBoxLayout()
    reachSlider = make_slider(0, 500)
    reachSlider.setTickInterval(2)
    reachSlider.setValue(1)
    reachSliderBox.addWidget(reachSlider)
    reachSlider.setValue(10)
    reachSliderLabel = QLabel(str(reachSlider.value()))
    reachSliderBox.addWidget(QLabel("reach"))
    reachSliderBox.addWidget(reachSliderLabel)

    def reach_value_changed(new_value):
        global blurred_img
        reachSliderLabel.setText(str(new_value))
        reach = new_value
        if reach%2 == 0:
            reach+=1
        if image_to_process is not None:
            blurred_img = cv.GaussianBlur(image_to_process, (reach, reach), 0)
        process_img()

    reachSlider.valueChanged.connect(reach_value_changed)

    slidersAndPictureBox.addLayout(minSliderBox)
    slidersAndPictureBox.addLayout(midSliderBox)
    slidersAndPictureBox.addLayout(maxSliderBox)
    slidersAndPictureBox.addLayout(reachSliderBox)
    slidersAndPictureBox.addStretch(1)
    label_imageDisplay = QLabel()
    label_imageDisplay.sizeHint()
    label_imageDisplay.setMinimumSize(1,1)
    slidersAndPictureBox.addWidget(label_imageDisplay)
    slidersAndPictureBox.addStretch(1)

    layout = QVBoxLayout()
    layout.addLayout(slidersAndPictureBox)

    buttonsBox = QHBoxLayout()
    openButton = QPushButton()
    openButton.setText("Open")
    openButton.clicked.connect(open_image)
    buttonsBox.addWidget(openButton)

    saveButoon = QPushButton()
    saveButoon.setText("Save")
    saveButoon.clicked.connect(save_image)
    buttonsBox.addWidget(saveButoon)

    plotButton = QPushButton()
    plotButton.setText("Show Threshold Plot")
    def show_plot():
        x = [0, 127, 255]
        y = [minSlider.value(), midSlider.value(), maxSlider.value()]

        A = np.matrix([[el ** i for i in range(3)] for el in x], dtype=np.float64)
        B = np.matrix(y, dtype=np.float64).T
        coef = np.linalg.solve(A, B).A1
        xs = np.arange(0,255.1,0.1)
        def eval(x):
            return coef[0] + x * coef[1] + (x ** 2) * coef[2]

        ys = eval(xs)
        fig, ax = plt.subplots()
        ax.plot(xs,ys)
        ax.set(ylim=(0,255))
        fig.show()

    plotButton.clicked.connect(show_plot)
    buttonsBox.addWidget(plotButton)



    layout.addLayout(buttonsBox)

    window.setLayout(layout)
    window.showMaximized()
    app.exec()






