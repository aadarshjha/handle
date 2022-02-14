import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split # Helps with organizing data for training
from sklearn.metrics import confusion_matrix # Helps present results as a confusion-matrix

# collect the data
def collect_data(): 
    imagepaths = []

    # Go through all the files and subdirectories inside a folder and save path to images inside list
    for root, dirs, files in os.walk(".", topdown=False): 
    for name in files:
        path = os.path.join(root, name)
        if path.endswith("png"): # We want only the images
        imagepaths.append(path)

    print(len(imagepaths)) # If > 0, then a PNG image was loaded