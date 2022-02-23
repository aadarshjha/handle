# parallelizing image capture
import re
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
import cv2 as cv

class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)

    def run(self): 
        self.ThreadActive = True
        Capture = cv.VideoCapture(0)
        while self.ThreadActive:
            ret, frame = Capture.read()
            if ret: 
                Image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                FlippedImage = cv.flip(Image, 1)
                ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640,480,Qt.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)
    
    def stop(self): 
        self.ThreadActive = False 
        self.quit()