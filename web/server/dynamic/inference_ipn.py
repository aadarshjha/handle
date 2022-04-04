# inference class
from cProfile import label
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
with open("static/labels/asl.json") as f:
    labels = json.load(f)

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
gpus = tf.config.experimental.list_physical_devices("GPU")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class InferenceIPN:
    def __init__(self, blob):
        self.blob = blob

    def processVideo(self):
        # convert blob string to object
        self.blob = json.loads(self.blob)
        bs64 = self.blob["blob"]

        # remove the metadata of the base64 string
        bs64 = bs64.split(",")[1]

        # save bs64 to a file 
        with open("./video.txt", "w") as f:
            f.write(bs64)
    
        bs64bytes = base64.b64decode(bs64)
        with open("./video.webm", "wb") as f:
            f.write(bs64bytes)

        return None

    # def preProcess(self):
    #     augmented_image = self.augment_single_image()
    #     asl_image = augmented_image

    #     model_asl = None

    #     # apply the correct model:
    #     if self.model == "cnn":
    #         model_asl = keras.models.load_model("static/model/asl/cnn.h5")
    #     elif self.model == "densenet":
    #         model_asl = keras.models.load_model("static/model/asl/densenet.h5")
    #     elif self.model == "densenet_pretrained":
    #         model_asl = keras.models.load_model(
    #             "static/model/asl/densenet_pretrained.h5"
    #         )
    #     elif self.model == "resnet":
    #         model_asl = keras.models.load_model("static/model/asl/resnet.h5")
    #     elif self.model == "resnet_pretrained":
    #         model_asl = keras.models.load_model("static/model/asl/resnet_pretrained.h5")
    #     elif self.model == "mobilenet":
    #         model_asl = keras.models.load_model("static/model/asl/mobilenet.h5")
    #     elif self.model == "mobilenet_pretrained":
    #         model_asl = keras.models.load_model(
    #             "static/model/asl/mobilenet_pretrained.h5"
    #         )
    #     else:
    #         print("Error: model not found")
    #         return None

    #     # predict the image
    #     prediction_asl = model_asl.predict(asl_image.reshape(1, 28, 28, 1))
    #     prediction_asl = np.argmax(prediction_asl, axis=1)
    #     prediction_asl = labels[str(prediction_asl[0])]

    #     return prediction_asl

    # def augment_single_image(self):
    #     gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    #     image = cv2.resize(gray, (28, 28))
    #     return image

    # def convert_to_b64(self, img):
    #     # convert a array to base64
    #     ret, png = cv2.imencode(".png", img)
    #     png_str = base64.b64encode(png)
    #     return png_str

    # def decode(self):
    #     new_image = None
    #     if isinstance(self.image, bytes):
    #         new_image = self.image.decode("utf-8")
    #     else:
    #         new_image = self.image
    #     imgdata = base64.b64decode(new_image)
    #     img = skimage.io.imread(imgdata, plugin="imageio")
    #     self.image = img
