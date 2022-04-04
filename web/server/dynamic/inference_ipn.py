# inference class
from cProfile import label
import os
from pickle import NONE
from pyexpat import model
import numpy as np
import cv2
import pandas as pd
import json
import base64
import subprocess
import torch 
import torchvision

import sys

# find somee labels -- later on. 
# # read ./labels/hgrd.json
# with open("static/labels/asl.json") as f:
#     labels = json.load(f)

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

class InferenceIPN:
    def __init__(self, blob):
        self.blob = blob

    def processFrames(self): 
        pass 

    def fetchFrames(self):
        # convert blob string to object
        self.blob = json.loads(self.blob)
        bs64 = self.blob["blob"]

        # remove the metadata of the base64 string
        bs64 = bs64.split(",")[1]

        # save bs64 to a file
        with open("./video.txt", "w") as f:
            f.write(bs64)

        bs64bytes = base64.b64decode(bs64)

        # save the video a file preprocsessing.webm, this is the video that
        # will be used for the preprocessing
        with open("./temp_out.webm", "wb") as f:
            f.write(bs64bytes)

        subprocess.call(["./dynamic/ffmpeg", "-i", "temp_out.webm", "out.mp4", "-y"])
        cap = cv2.VideoCapture(sys.path[0] + "/out.mp4")
        success, image = cap.read()
        frames = []
        while success:
            frames.append(image)
            success, image = cap.read()

        print(len(frames))
        return frames

    def rejectionCriterion(self, test_num): 
        if test_num < 10: 
            raise Exception("rejectionCriterionError: The video needs be at 30 frames.")
        else: 
            print("Works")