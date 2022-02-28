from flask import Flask, request
from static.process import Process
from static.inference import Inference
from flask_cors import CORS, cross_origin
import json

app = Flask(__name__)
# CORS(app, support_credentials=True)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config["CORS_HEADERS"] = "Content-Type"

# accept a JSON object
@app.route("/static/cnn", methods=["POST", "GET"])
@cross_origin(supports_credentials=True)
def index():
    if request.method == "POST":
        print("hello world")
        fetched_image = request.get_json()["imageSrc"]
        augmented_image = Inference(fetched_image)
        augmented_image.decode()
        augmented_single_image = augmented_image.augment_single_image()
        return json.dumps(augmented_single_image.tolist())


if __name__ == "__main__":
    app.run(debug=True)
