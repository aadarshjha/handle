# inference class

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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
import json
from sklearn.model_selection import train_test_split

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
gpus= tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class Inference: 
    def __init__(self, image):
        self.image = image
    
    def preProcess(self): 
        pass
