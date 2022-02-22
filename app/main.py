from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout
from PyQt6.QtGui import QPalette, QColor
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
        return Color("Blue")

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