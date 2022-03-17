import os
from pickle import NONE
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import yaml
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
import json
from sklearn.model_selection import train_test_split

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def create_model(mode="CNN"):
    model = None
    if mode == "CNN":
        model = Sequential()
        model.add(Conv2D(32, (5, 5), activation="relu", input_shape=(120, 320, 1)))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(10, activation="softmax"))
    elif mode == "CNN_PRETRAINED":
        pass
    elif mode == "RESNET":
        model = keras.applications.resnet.ResNet50(
            include_top=False, weights=None, input_shape=(120, 320, 1)
        )
    elif mode == "RESNET_PRETRAINED":
        model = keras.applications.resnet.ResNet50(
            include_top=False, weights="imagenet", input_shape=(120, 320, 1)
        )
    elif mode == "MOBILENET":
        model = keras.applications.mobilenet.MobileNet(
            include_top=False, weights=None, input_shape=(120, 320, 1)
        )
    elif mode == "MOBILENET_PRETRAINED":
        model = keras.applications.mobilenet.MobileNet(
            include_top=False, weights="imagenet", input_shape=(120, 320, 1)
        )
    elif mode == "DENSENET":
        model = keras.applications.densenet.DenseNet121(
            include_top=False, weights=None, input_shape=(120, 320, 1)
        )
    elif mode == "DENSENET_PRETRAINED":
        model = keras.applications.densenet.DenseNet121(
            include_top=False, weights="imagenet", input_shape=(120, 320, 1)
        )
    else:
        # throw an error to the user
        raise Exception("Invalid model type")

    return model
