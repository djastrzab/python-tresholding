from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import cv2 as cv
import numpy as np
import sys


def parabola_gaussian_img(img, min_threshold, mid_threshold,max_threshold, reach):
    if reach%2 == 0:
        reach += 1

    blurred_grayscale = cv.GaussianBlur(img, (reach, reach), 0)

    x = [0, 127, 255]
    y = [min_threshold, mid_threshold, max_threshold]

    A = np.matrix([[el**i for i in range(3)] for el in x], dtype=np.float64)
    B = np.matrix(y, dtype=np.float64).T
    coef = np.linalg.solve(A, B).A1

    def threshold(x):
        return coef[0]+x*coef[1]+(x**2)*coef[2]

    return np.array((img >= threshold(np.array(blurred_grayscale, dtype=np.float64)))*255, dtype=np.uint8)


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
    result_img = None

    def process_img():
        global result_img
        if image_to_process is not None:
            print(minSlider.value(), midSlider.value(),maxSlider.value(), reachSlider.value())
            result_img = parabola_gaussian_img(image_to_process, minSlider.value(), midSlider.value(),maxSlider.value(), reachSlider.value())
            h, w = result_img.shape
            label_imageDisplay.setPixmap(QtGui.QPixmap.fromImage(QImage(result_img.data, w, h, w, QImage.Format.Format_Grayscale8)))

    def open_image():
        global image_to_process
        filename = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
        if filename == "":
            return 
        image_to_process = cv.imread(filename, cv.IMREAD_GRAYSCALE)
        print(image_to_process.data)
        h, w = image_to_process.shape
        label_imageDisplay.setPixmap(QtGui.QPixmap.fromImage(QImage(image_to_process.data, w, h, w, QImage.Format.Format_Grayscale8)))

    def save_image():
        if result_img is None:
            return
        filename = QFileDialog.getSaveFileName(filter="Image (*.tif)")[0]
        print(filename)
        cv.imwrite(filename, result_img)

    minSliderBox = QVBoxLayout()
    minSlider = make_slider(0, 255)
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
    reachSliderLabel = QLabel(str(reachSlider.value()))
    reachSliderBox.addWidget(QLabel("reach"))
    reachSliderBox.addWidget(reachSliderLabel)

    def reach_value_changed(new_value):
        reachSliderLabel.setText(str(new_value))
        process_img()

    reachSlider.valueChanged.connect(reach_value_changed)

    slidersAndPictureBox.addLayout(minSliderBox)
    slidersAndPictureBox.addLayout(midSliderBox)
    slidersAndPictureBox.addLayout(maxSliderBox)
    slidersAndPictureBox.addLayout(reachSliderBox)
    slidersAndPictureBox.insertStretch(4,2)

    label_imageDisplay = QLabel()
    slidersAndPictureBox.addWidget(label_imageDisplay)

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

    layout.addLayout(buttonsBox)

    window.setLayout(layout)
    window.showMaximized()
    app.exec()






