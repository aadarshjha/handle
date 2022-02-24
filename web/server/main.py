from flask import Flask, request
from static.process import Process
from static.inference import Inference
from flask_cors import CORS, cross_origin


app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

# accept a JSON object
@app.route("/static/cnn", methods=["POST", "GET"])
def index():
    data = request.get_json()
    image = Process(data["imageSrc"]).readb64()
    if request.method == "POST":
        return ""
    elif request.method == "GET":
        preprocessed = Inference(image).augment_single_image()
        preprocessed = Inference(image).convert_to_b64()
        return preprocessed
    else:
        # POST error 405
        print("Method not allowed")


# return a JSON object
@app.route("/", methods=["GET"])
def label():
    return {"label": "hello"}


if __name__ == "__main__":
    app.run(debug=True)
