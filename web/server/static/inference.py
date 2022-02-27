# inference class

import os
from pickle import NONE
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import pandas as pd
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
import json
import base64

# read ./labels/hgrd.json
with open("static/labels/hgrd.json") as f:
    labels = json.load(f)

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
gpus = tf.config.experimental.list_physical_devices("GPU")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class Inference:
    def __init__(self, image):
        self.image = image

    def preProcess(self):
        augmented_image = self.augment_single_image()
        model = keras.models.load_model("static/model/5.h5")
        prediction = model.predict(augmented_image.reshape(1, 120, 320, 1))
        return labels[str(prediction[0])]

    def augment_single_image(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(gray, (320, 120))
        return img

    def convert_to_b64(self):
        return base64.b64encode(self.image)
