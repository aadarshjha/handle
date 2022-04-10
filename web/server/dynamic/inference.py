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
from PIL import Image
from dynamic.transform_utils import (
    TemporalCenterCrop,
    Compose,
    ToTensor,
    Normalize,
    Scale,
    CenterCrop,
)
from dynamic.network import *
import sys
import yaml

# find somee labels -- later on.
# # read ./labels/hgrd.json
# with open("static/labels/asl.json") as f:
#     labels = json.load(f)

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


classes = ["No Gesture", "Pointing With One Finger", "Double Click With One Finger"]


class Config:
    def __init__(self, in_dict: dict):
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
                setattr(
                    self, key, [DictObj(x) if isinstance(x, dict) else x for x in val]
                )
            else:
                setattr(self, key, DictObj(val) if isinstance(val, dict) else val)


class Inference:
    def __init__(self, blob, mode):
        self.blob = blob
        self.mode = mode

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
        return frames

    def preProcess(self, frames):
        # convert opencv image to PIL
        pil_frames = []

        for element in frames:
            pil_frames.append(Image.fromarray(cv2.cvtColor(element, cv2.COLOR_BGR2RGB)))

        temporalTransforms = Compose([TemporalCenterCrop(32)])
        spatialTransforms = Compose(
            [
                Scale(112),
                CenterCrop(112),
                ToTensor(1),
                Normalize(
                    [114.7748, 107.7354, 99.475], [38.7568578, 37.88248729, 40.02898126]
                ),
            ]
        )

        # create a list of indices for the frames
        indices = list(range(len(pil_frames)))
        indices = temporalTransforms(indices)

        clip = []

        spatialTransforms.randomize_parameters()

        # load up the frames from the clip
        for i in indices:
            clip.append(spatialTransforms(pil_frames[i]))

        dim = clip[0].size()[-2:]
        clip = torch.cat(clip, 0).view((32, -1) + dim).permute(1, 0, 2, 3)
        return clip

    def inference(self, clip):

        prediction_list = []
        config_dict = {}

        if self.mode == "resnext":

            try:
                with open("./dynamic/configs/ego_config.yaml", "r") as f:
                    user_config = yaml.safe_load(f)
                    config_dict.update(user_config)
            except Exception:
                print("Error reading config file")
                exit(1)

            config_dict["device"] = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
            config = Config(config_dict)
            model, _ = load_resnext101(config, device=config.device)

            # pass in clip to model
            model.eval()

            clip = clip.unsqueeze(0)
            clip = clip.to(config.device)

            with torch.no_grad():
                output = model(clip)
                _, pred = output.topk(1, 1)
                idx = pred.squeeze().cpu().numpy()

            prediction_list.append(classes[idx])

            try:
                with open("./dynamic/configs/ipn_config.yaml", "r") as f:
                    user_config = yaml.safe_load(f)
                    config_dict.update(user_config)
            except Exception:
                print("Error reading config file")
                exit(1)

            config = Config(config_dict)
            model, _ = load_resnext101(config, device=config.device)

            # pass in clip to model
            model.eval()

            with torch.no_grad():
                output = model(clip)
                _, pred = output.topk(1, 1)
                idx = pred.squeeze().cpu().numpy()

            prediction_list.append(classes[idx])
            print(prediction_list)
            return prediction_list
        elif self.mode == "lstm":
            try:
                with open("./dynamic/configs/lstm_config.yaml", "r") as f:
                    user_config = yaml.safe_load(f)
                    config_dict.update(user_config)
            except Exception:
                print("Error reading config file")
                exit(1)

            config_dict["device"] = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )

            config = Config(config_dict)
            model, _ = load_cnn_lstm(config, device=config.device)

            # pass in clip to model
            model.eval()

            clip = clip.unsqueeze(0)
            clip = clip.to(config.device)

            with torch.no_grad():
                output = model(clip)
                _, pred = output.topk(1, 1)
                idx = pred.squeeze().cpu().numpy()

            prediction_list.append(classes[idx])
            return prediction_list
        elif self.mode == "timesformer":
            try:
                with open("./dynamic/configs/timesformer_config.yaml", "r") as f:
                    user_config = yaml.safe_load(f)
                    config_dict.update(user_config)
            except Exception:
                print("Error reading config file")
                exit(1)

            config_dict["device"] = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )

            config = Config(config_dict)
            model, _ = load_timesformer(config, device=config.device)

            # pass in clip to model
            model.eval()

            clip = clip.unsqueeze(0)
            clip = clip.to(config.device)

            with torch.no_grad():
                output = model(clip)
                _, pred = output.topk(1, 1)
                idx = pred.squeeze().cpu().numpy()

            print(prediction_list)
            prediction_list.append(classes[idx])
            return prediction_list

    def rejectionCriterion(self, test_num):
        return test_num >= 32
