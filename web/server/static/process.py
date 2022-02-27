import cv2 as cv
import base64
import numpy as np

# class to process images
class Process:
    # constructor
    def __init__(self, imagebytes):
        self.imagebytes = imagebytes

    def readb64(self):
        encoded_data = self.imagebytes.split(",")[1]
        nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
        img = cv.imdecode(nparr, cv.IMREAD_COLOR)
        return img
