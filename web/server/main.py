from flask import Flask, request
from static.process import Process
from static.inference import Inference
from flask_cors import CORS, cross_origin
import jsonify

app = Flask(__name__)
CORS(app, support_credentials=True)
app.config["CORS_HEADERS"] = "Content-Type"

# accept a JSON object
@app.route("/static/cnn", methods=["POST", "GET"])
@cross_origin(supports_credentials=True)
def index():
    if request.method == "POST":
        # print the posted data
        # print(request.get_json())
        fetched_image = request.get_json()["imageSrc"]
        augmented_image = Inference(fetched_image).augment_single_image()

        return {"hello": "world"}


# return a JSON object
@app.route("/", methods=["GET"])
def label():
    return {"label": "hello"}


if __name__ == "__main__":
    app.run(debug=True)
