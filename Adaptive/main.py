from PyQt5 import QtWidgets
from firstLayout import Ui_MainWindow
import sys

if __name__ == '__main__':
    #cv.imwrite("zamek_parabolic_faster.tif", parabola_gaussian_improved("original.jpg", 100, 230, 31))
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(window)
    window.show()
    sys.exit(app.exec_())