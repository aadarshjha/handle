from lib2to3.pytree import convert
from flask import Flask, request
from static.process import Process
from static.inference import Inference
from flask_cors import CORS, cross_origin
import json
import base64

import cv2 as cv

app = Flask(__name__)
# CORS(app, support_credentials=True)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config["CORS_HEADERS"] = "Content-Type"

# accept a JSON object
@app.route("/static/inference", methods=["POST", "GET"])
@cross_origin(supports_credentials=True)
def index():
    if request.method == "POST":
        print(request.get_json()["model"])
        print(request.get_json()["mode"])

        fetched_image = request.get_json()["imageSrc"]
        model = request.get_json()["model"]
        mode = request.get_json()["mode"]
        augmented_image = Inference(fetched_image, model, mode)
        augmented_image.decode()
        augmented_single_image = augmented_image.augment_single_image()

        # convert augmented_single_image to base64
        augmented_single_image_b64 = augmented_image.convert_to_b64(
            augmented_single_image
        )

        return json.dumps(
            {
                "image": augmented_single_image_b64.decode("utf-8"),
                "prediction": augmented_image.preProcess(),
            }
        )


if __name__ == "__main__":
    app.run(debug=True)
