# inference class

import os
from pickle import NONE
from pyexpat import model
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
import skimage.io


# read ./labels/hgrd.json
with open("static/labels/hgrd.json") as f:
    labels = json.load(f)

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
gpus = tf.config.experimental.list_physical_devices("GPU")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class Inference:
    def __init__(self, image, model, mode):
        self.image = image
        self.model = model
        self.mode = mode

    def preProcess(self):
        augmented_image = self.augment_single_image()

        model = None

        # apply the correct model:
        if self.model == "cnn":
            model = keras.models.load_model("static/model/cnn.h5")
        elif self.model == "densenet":
            model = keras.models.load_model("static/model/densenet.h5")
        elif self.model == "densenet_pretrained":
            model = keras.models.load_model("static/model/densenet_pretrained.h5")
        elif self.model == "resnet":
            model = keras.models.load_model("static/model/resnet.h5")
        elif self.model == "resnet_pretrained":
            model = keras.models.load_model("static/model/resnet_pretrained.h5")
        elif self.model == "mobilenet":
            model = keras.models.load_model("static/model/mobilenet.h5")
        elif self.model == "mobilenet_pretrained":
            model = keras.models.load_model("static/model/mobilenet_pretrained.h5")
        else:
            print("Error: model not found")
            return None

        prediction = model.predict(augmented_image.reshape(1, 120, 320, 1))
        prediction = np.argmax(prediction, axis=1)
        prediction = labels[str(prediction[0])]
        return prediction

    def augment_single_image(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(gray, (320, 120))
        return img

    def convert_to_b64(self, img):
        # convert a array to base64
        ret, png = cv2.imencode(".png", img)
        png_str = base64.b64encode(png)
        return png_str

    def decode(self):
        new_image = None
        if isinstance(self.image, bytes):
            new_image = self.image.decode("utf-8")
        else:
            new_image = self.image
        imgdata = base64.b64decode(new_image)
        img = skimage.io.imread(imgdata, plugin="imageio")
        self.image = img
