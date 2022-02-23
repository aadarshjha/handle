from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
import json
from color import *
from camera import * 
from worker1 import *
import cv2 as cv

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

        self.FeedLabel = QLabel()

        layout.addWidget(self.FeedLabel)
        layout.addWidget(self.right())

        self.Worker1 = Worker1()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        self.Worker1.start()

        widget = QWidget()
        widget.setLayout(layout)

        # removes odd spacing
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.setCentralWidget(widget)


    def ImageUpdateSlot(self, Image): 
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))
    
    # def CancelFee


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

    # def create_camera(self):
    #     # create a video feed 

    #     self.available_cameras = QCameraInfo.availableCameras()

    #     if not self.available_cameras: 
    #         # return a warning
    #         return QLabel('No camera detected')

    #     self.camera = QWidget(self)
    #     self.camera.setAutoFillBackground(True)
    #     self.camera.setFixedSize(int(WIDTH * 0.5), int(HEIGHT * 0.5))
    #     self.camera.setStyleSheet("background-color: blue")

        # self.camera.setWidget(self.viewfinder)

                # making it central widget of main window
        # self.setCentralWidget(self.viewfinder)

        # self.camera_test = QCamera(self.available_cameras[0])
        # self.camera_test.setViewfinder(self.viewfinder)
        # self.camera_test.setCaptureMode(QCamera.CaptureStillImage)

        # # if any error occur show the alert
        # self.camera_test.error.connect(lambda: self.alert(self.camera.errorString()))
  
        # # start the camera
        # self.camera_test.start()


        

        # return self.camera

    def right(self):
        right_widget = QWidget(self)
        right_widget.setAutoFillBackground(True)
        right_widget.setFixedSize(int(WIDTH * 0.6), HEIGHT)
        right_widget.setStyleSheet("background-color: red")
        return right_widget


