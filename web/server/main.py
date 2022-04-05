from lib2to3.pytree import convert
from flask import Flask, request
from static.process import Process
from static.inference_hgr import InferenceHGR
from static.inference_asl import InferenceASL
from flask_cors import CORS, cross_origin
from dynamic.inference import Inference

import json
import cv2 as cv

app = Flask(__name__)
# CORS(app, support_credentials=True)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config["CORS_HEADERS"] = "Content-Type"

# accept a JSON object
@app.route("/dynamic/cnn", methods=["POST", "GET"])
@cross_origin(supports_credentials=True)
def dynamic_index():
    if request.method == "POST":

        json_obj = request.get_json()
        blob = json_obj["videoSrc"]
        mode = json_obj["model"]

        inferenceClass = Inference(blob, mode)
        frames = inferenceClass.fetchFrames()

        if not inferenceClass.rejectionCriterion(len(frames)):
            return json.dumps({"Error": "Video Is Rejected, Needs To Be More Long"})

        if mode == "resnext":
            clip = inferenceClass.preProcess(frames)
            inference = inferenceClass.inference(clip)
            return json.dumps(
                {
                    "EgoGesture": {"prediction": inference[0]},
                    "IPN": {"prediction": inference[1]},
                }
            )
        elif mode == "lstm":
            pass
        elif mode == "timesformer":
            pass
        else:
            return json.dumps({"Error": "Model Not Found"})


# accept a JSON object
@app.route("/static/cnn", methods=["POST", "GET"])
@cross_origin(supports_credentials=True)
def static_index():
    if request.method == "POST":
        fetched_image = request.get_json()["imageSrc"]
        model = request.get_json()["model"]
        mode = request.get_json()["mode"]

        # for the HGR dataset
        augmented_image_hgr = InferenceHGR(fetched_image, model, mode)
        augmented_image_hgr.decode()
        augmented_single_image_hgr = augmented_image_hgr.augment_single_image()

        # convert augmented_single_image to base64
        augmented_single_image_b64_hgr = augmented_image_hgr.convert_to_b64(
            augmented_single_image_hgr
        )

        augmented_image_asl = InferenceASL(fetched_image, model, mode)
        augmented_image_asl.decode()
        augmented_single_image_asl = augmented_image_asl.augment_single_image()

        # convert augmented_single_image to base64
        augmented_single_image_b64_asl = augmented_image_asl.convert_to_b64(
            augmented_single_image_asl
        )

        return json.dumps(
            {
                "HGR": {
                    "image": augmented_single_image_b64_hgr.decode("utf-8"),
                    "prediction": augmented_image_hgr.preProcess(),
                },
                "ASL": {
                    "image": augmented_single_image_b64_asl.decode("utf-8"),
                    "prediction": augmented_image_asl.preProcess(),
                },
            }
        )


if __name__ == "__main__":
    app.run(debug=True)
