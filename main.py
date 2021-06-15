from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
import imutils
import ntpath


def parabola_gaussian_img(img, blurred_img, min_threshold, mid_threshold, max_threshold):
    x = [0, 127, 255]
    y = [min_threshold, mid_threshold, max_threshold]

    A = np.matrix([[el ** i for i in range(3)] for el in x], dtype=np.float64)
    B = np.matrix(y, dtype=np.float64).T
    coef = np.linalg.solve(A, B).A1

    def threshold(x):
        return coef[0] + x * coef[1] + (x ** 2) * coef[2]

    return np.array((img >= np.array([threshold(val) for val in range(0, 256)])[blurred_img]) * 255, dtype=np.uint8)

    # return np.array((img >= threshold(np.array(blurred_img, dtype=np.float64))) * 255, dtype=np.uint8)


def gaussian(img, C, reach):
    if reach % 2 == 0:
        reach += 1
    output_img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, reach, C)
    return output_img


def cubic_gaussian(img, blurred_img, min_threshold, mid_threshold, max_threshold):
    h = 255 / 2
    x = [0, 255 / 2, 255]
    y = [min_threshold, mid_threshold, max_threshold]
    A = np.matrix(
        [[2 * h, h, 0],
         [h, 4 * h, h],
         [0, h, 2 * h]]
        , dtype=np.float64).T

    def delta(i):
        return (y[i] - y[i - 1]) / h

    b = np.matrix([delta(1), delta(2) - delta(1), -delta(2)], dtype=np.float64).T
    sigmas = np.linalg.solve(A, b).A1

    def threshold(val):
        if val < 127.5:
            return (
                y[0]
                + ((y[0 + 1] - y[0]) / h - h * (sigmas[0 + 1] + 2 * sigmas[0])) * (val - x[0])
                + (3 * sigmas[0]) * (val - x[0]) ** 2
                + ((sigmas[0 + 1] - sigmas[0]) / h) * (val - x[0]) ** 3
            )
        return (
            y[1]
            + ((y[1 + 1] - y[1]) / h - h * (sigmas[1 + 1] + 2 * sigmas[1])) * (val - x[1])
            + (3 * sigmas[1]) * (val - x[1]) ** 2
            + ((sigmas[1 + 1] - sigmas[1]) / h) * (val - x[1]) ** 3
        )

    return np.array((img >= np.array([threshold(val) for val in range(0, 256)])[blurred_img]) * 255, dtype=np.uint8)


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
    mode = 1


    def resize_img(image):
        screen_resolution = QApplication.primaryScreen()
        height = screen_resolution.size().height() - 140
        view_image = imutils.resize(image, height=height)
        return view_image


    def process_img():
        global result_img
        if image_to_process is not None:
            # print(minSlider.value(), midSlider.value(),maxSlider.value(), reachSlider.value())
            if mode == 1:
                result_img = parabola_gaussian_img(image_to_process, blurred_img, minSlider.value(), midSlider.value(),
                                                   maxSlider.value())
            if mode == 2:
                result_img = gaussian(image_to_process, cSlider.value(), reachSlider.value())
            if mode == 3:
                result_img = cubic_gaussian(image_to_process, blurred_img, minSlider.value(), midSlider.value(),
                                            maxSlider.value())
            view_image = resize_img(result_img)
            h, w = view_image.shape
            pix = QPixmap.fromImage(QImage(view_image.data, w, h, w, QImage.Format.Format_Grayscale8))
            label_imageDisplay.setPixmap(pix)


    def open_image():
        global image_to_process
        global blurred_img
        reach = reachSlider.value()
        if reach % 2 == 0:
            reach += 1
        filename = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
        if filename == "":
            return
        image_to_process = cv.imread(filename, cv.IMREAD_GRAYSCALE)
        blurred_img = cv.GaussianBlur(image_to_process, (reach, reach), 0)
        view_image = resize_img(image_to_process)
        h, w = view_image.shape
        pix = QPixmap.fromImage(QImage(view_image.data, w, h, w, QImage.Format.Format_Grayscale8))
        label_imageDisplay.setPixmap(pix)


    def batch_mode():
        global mode
        reach = reachSlider.value()
        if reach % 2 == 0:
            reach += 1
        filenames = QFileDialog.getOpenFileNames(filter="Image (*.*)")[0]
        directory = QFileDialog.getExistingDirectory()
        if filenames == "" or directory == "":
            return
        for file in filenames:
            image = cv.imread(file, cv.IMREAD_GRAYSCALE)
            blurred_image = cv.GaussianBlur(image, (reach, reach), 0)
            if mode == 1:
                result = parabola_gaussian_img(image, blurred_image, minSlider.value(), midSlider.value(),
                                               maxSlider.value())
            if mode == 2:
                result = gaussian(image, cSlider.value(), reachSlider.value())
            if mode == 3:
                result = cubic_gaussian(image, blurred_image, minSlider.value(), midSlider.value(),
                                        maxSlider.value())
            filename=ntpath.basename(file)
            cv.imwrite(os.path.join(directory, filename), result)



    def save_image():
        if result_img is None:
            return
        filename = QFileDialog.getSaveFileName(filter="Tagged Image file (*.tif);;Portable Network Graphics (*.png);; "
                                                      "Joint Photographic Experts Group (*.jpg);; other *.*")[0]

        if filename == "":
            return
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
        if reach % 2 == 0:
            reach += 1
        if image_to_process is not None:
            blurred_img = cv.GaussianBlur(image_to_process, (reach, reach), 0)
        process_img()


    reachSlider.valueChanged.connect(reach_value_changed)

    slidersParabolaBox = QHBoxLayout()
    slidersParabolaBox.addLayout(minSliderBox)
    slidersParabolaBox.addLayout(midSliderBox)
    slidersParabolaBox.addLayout(maxSliderBox)

    slidersLinearBox = QHBoxLayout()

    cSliderBox = QVBoxLayout()
    cSlider = make_slider(-20, 20)
    cSlider.setValue(0)
    cSliderBox.addWidget(cSlider)
    cSliderBox.addWidget(QLabel("C"))
    cSliderLabel = QLabel(str(cSlider.value()))
    cSliderBox.addWidget(cSliderLabel)


    def c_val_changed(new_val):
        cSliderLabel.setText(str(new_val))
        process_img()


    cSlider.valueChanged.connect(c_val_changed)
    slidersLinearBox.addLayout(cSliderBox)

    slidersLinearFrame = QFrame()
    slidersParabolaFrame = QFrame()
    slidersReachFrame = QFrame()
    slidersLinearFrame.setLayout(slidersLinearBox)
    slidersParabolaFrame.setLayout(slidersParabolaBox)
    slidersReachFrame.setLayout(reachSliderBox)

    slidersAndPictureBox.addWidget(slidersParabolaFrame)
    slidersAndPictureBox.addWidget(slidersLinearFrame)
    slidersAndPictureBox.addWidget(slidersReachFrame)
    slidersLinearFrame.hide()

    slidersAndPictureBox.addLayout(reachSliderBox)
    slidersAndPictureBox.addStretch(1)
    label_imageDisplay = QLabel()
    label_imageDisplay.sizeHint()
    label_imageDisplay.setMinimumSize(1, 1)
    slidersAndPictureBox.addWidget(label_imageDisplay)
    slidersAndPictureBox.addStretch(1)

    layout = QVBoxLayout()
    layout.addLayout(slidersAndPictureBox)

    buttonsBox = QHBoxLayout()
    openButton = QPushButton()
    openButton.setText("Otwórz")
    openButton.clicked.connect(open_image)
    buttonsBox.addWidget(openButton)

    saveButoon = QPushButton()
    saveButoon.setText("Zapisz")
    saveButoon.clicked.connect(save_image)
    buttonsBox.addWidget(saveButoon)

    batchButoon = QPushButton()
    batchButoon.setText("Tryb wsadowy")
    batchButoon.clicked.connect(batch_mode)
    buttonsBox.addWidget(batchButoon)

    plotButton = QPushButton()
    plotButton.setText("Wykres progowania")


    def show_plot():
        fig, ax = plt.subplots()
        xs = np.arange(0, 255.1, 0.1)
        ax.set(ylim=(0, 255))

        if mode == 1:
            x = [0, 127, 255]
            y = [minSlider.value(), midSlider.value(), maxSlider.value()]

            A = np.matrix([[el ** i for i in range(3)] for el in x], dtype=np.float64)
            B = np.matrix(y, dtype=np.float64).T
            coef = np.linalg.solve(A, B).A1

            def eval(x):
                return coef[0] + x * coef[1] + (x ** 2) * coef[2]

            ys = eval(xs)
            ax.plot(xs, ys)
        if mode == 2:
            ys = xs - cSlider.value()
            ax.plot(xs, ys)
        if mode == 3:
            h = 255 / 2
            x = [0, 255 / 2, 255]
            y = [minSlider.value(), midSlider.value(), maxSlider.value()]
            A = np.matrix(
                [[2 * h, h, 0],
                 [h, 4 * h, h],
                 [0, h, 2 * h]]
                , dtype=np.float64).T

            def delta(i):
                return (y[i] - y[i - 1]) / h

            b = np.matrix([delta(1), delta(2) - delta(1), -delta(2)], dtype=np.float64).T
            sigmas = np.linalg.solve(A, b).A1

            def threshold(val):
                s1 = (y[0]
                      + ((y[0 + 1] - y[0]) / h - h * (sigmas[0 + 1] + 2 * sigmas[0])) * (val - x[0])
                      + (3 * sigmas[0]) * (val - x[0]) ** 2
                      + ((sigmas[0 + 1] - sigmas[0]) / h) * (val - x[0]) ** 3
                      ) * (val < 127.5)
                s2 = (y[1]
                      + ((y[1 + 1] - y[1]) / h - h * (sigmas[1 + 1] + 2 * sigmas[1])) * (val - x[1])
                      + (3 * sigmas[1]) * (val - x[1]) ** 2
                      + ((sigmas[1 + 1] - sigmas[1]) / h) * (val - x[1]) ** 3
                      ) * (val >= 127.5)
                return s1 + s2

            ys = threshold(xs)
            ax.plot(xs, ys)

        fig.show()


    plotButton.clicked.connect(show_plot)
    buttonsBox.addWidget(plotButton)

    chmodButton = QPushButton()
    chmodButton.setText("Change Mode")


    def chmod():
        global mode
        if mode == 1:
            mode = 2
            slidersParabolaFrame.hide()
            slidersLinearFrame.show()
        else:
            mode = 1
            slidersParabolaFrame.show()
            slidersLinearFrame.hide()
        print(mode)


    chmodButton.clicked.connect(chmod)
    # buttonsBox.addWidget(chmodButton)

    parabolaButton = QPushButton()
    parabolaButton.setText("Tryb paraboli")


    def change_mode_to_parabolic():
        global mode
        mode = 1
        slidersParabolaFrame.show()
        slidersLinearFrame.hide()
        process_img()


    parabolaButton.clicked.connect(change_mode_to_parabolic)

    linearButton = QPushButton()
    linearButton.setText("Tryb linowy")


    def change_mode_to_linear():
        global mode
        mode = 2
        slidersParabolaFrame.hide()
        slidersLinearFrame.show()
        process_img()


    linearButton.clicked.connect(change_mode_to_linear)

    cubicSplineButton = QPushButton()
    cubicSplineButton.setText("Tryb szesciennych funkcji sklejanych")


    def change_mode_to_cubic_spline():
        global mode
        mode = 3
        slidersParabolaFrame.show()
        slidersLinearFrame.hide()
        process_img()


    cubicSplineButton.clicked.connect(change_mode_to_cubic_spline)

    buttonsBox.addWidget(parabolaButton)
    buttonsBox.addWidget(linearButton)
    buttonsBox.addWidget(cubicSplineButton)

    layout.addLayout(buttonsBox)

    window.setLayout(layout)
    window.showMaximized()
    app.exec()
