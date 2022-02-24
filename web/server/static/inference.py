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

# read ./labels/hgrd.json
with open('static/labels/hgrd.json') as f:
    labels = json.load(f)

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
gpus= tf.config.experimental.list_physical_devices('GPU')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class Inference: 
    def __init__(self, image):
        self.image = image
    
    def preProcess(self): 
        # save the image to a file
        augmented_image = self.augment_single_image(self.image)
        # inference the prediction
        model = keras.models.load_model('static/model/5.h5')
        prediction = model.predict(augmented_image.reshape(1, 120, 320, 1))

        # get the prediction value
        prediction = np.argmax(prediction, axis=1)
        # get the prediction label
        
        return labels[str(prediction[0])]

    def augment_single_image(self, image): 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(gray, (320, 120))
        return img

