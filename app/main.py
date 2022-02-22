from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtMultimedia import *
from PyQt6.QtMultimediaWidgets import *
import sys
import json
from color import *

# read ./version.json
with open('./version.json', 'r') as f:
    version = json.load(f)

WIDTH = 1250
HEIGHT = 820 

class MainWindow(QMainWindow): 
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Handle: {} | {}".format(version.get('version'), 'Development' if version.get('development') else 'Stable'))

        # set fixed size of window
        self.setFixedSize(1250, 820)
        self.setLayout()

        self.available_devices = QMediaDevices.videoInputs()

        if not self.available_devices: 
            QMessageBox.warning(self, "No camera found", "No camera found")
            sys.exit(1)

        # print(self.available_cameras)
    
    def setLayout(self):

        # horizontal layout
        layout = QHBoxLayout()

        layout.addWidget(self.left())
        layout.addWidget(self.right())

        widget = QWidget()
        widget.setLayout(layout)

        # removes odd spacing
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.setCentralWidget(widget)

    def remove_spacing(self, layout): 
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

    def left(self): 
        left_widget = QWidget(self)
        left_widget.setAutoFillBackground(True)
        # set width to 40% and height to 100% of window
        left_widget.setFixedSize(int(WIDTH * 0.4), HEIGHT)
        # left_widget.setStyleSheet("background-color: blue")

        layout = QVBoxLayout()

        layout.addWidget(self.create_camera())
        layout.addWidget(Color('Green'))

        # remove margin and spacing
        self.remove_spacing(layout)

        left_widget.setLayout(layout)

        return left_widget

    def create_camera(self):
        # create a video feed 
        camera = QWidget(self)
        camera.setAutoFillBackground(True)
        camera.setFixedSize(int(WIDTH * 0.5), int(HEIGHT * 0.5))
        camera.setStyleSheet("background-color: blue")

        return camera

    def right(self):
        right_widget = QWidget(self)
        right_widget.setAutoFillBackground(True)
        right_widget.setFixedSize(int(WIDTH * 0.6), HEIGHT)
        right_widget.setStyleSheet("background-color: red")
        return right_widget

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec()