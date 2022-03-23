# inference class
import os
from pickle import NONE
from pyexpat import model
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import pandas as pd
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
        hgr_image = augmented_image

        model = None

        # apply the correct model:
        if self.model == "cnn":
            model_hgr = keras.models.load_model("static/model/hgr/cnn.h5")
            # model_asl = keras.models.load_model("static/model/asl/cnn.h5")
        elif self.model == "densenet":
            model_hgr = keras.models.load_model("static/model/hgr/densenet.h5")
            # model_asl = keras.models.load_model("static/model/asl/densenet.h5")
        elif self.model == "densenet_pretrained":
            model_hgr = keras.models.load_model(
                "static/model/hgr/densenet_pretrained.h5"
            )
            # model_asl = keras.models.load_model(
            #     "static/model/asl/densenet_pretrained.h5"
            # )
        elif self.model == "resnet":
            model_hgr = keras.models.load_model("static/model/hgr/resnet.h5")
            # model_asl = keras.models.load_model("static/model/asl/resnet.h5")
        elif self.model == "resnet_pretrained":
            model_hgr = keras.models.load_model("static/model/hgr/resnet_pretrained.h5")
            # model_asl = keras.models.load_model("static/model/asl/resnet_pretrained.h5")
        elif self.model == "mobilenet":
            model_hgr = keras.models.load_model("static/model/hgr/mobilenet.h5")
            # model_asl = keras.models.load_model("static/model/asl/mobilenet.h5")
        elif self.model == "mobilenet_pretrained":
            model_hgr = keras.models.load_model(
                "static/model/hgr/mobilenet_pretrained.h5"
            )
            # model_asl = keras.models.load_model(
            #     "static/model/asl/mobilenet_pretrained.h5"
            # )
        else:
            print("Error: model not found")
            return None

        prediction_hgr = model_hgr.predict(hgr_image.reshape(1, 120, 320, 1))
        prediction_hgr = np.argmax(prediction_hgr, axis=1)
        prediction_hgr = labels[str(prediction_hgr[0])]

        return prediction_hgr

    def augment_single_image(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        img_hgr = cv2.resize(gray, (320, 120))
        return img_hgr

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
