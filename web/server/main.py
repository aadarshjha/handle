from flask import Flask, request
from static.process import Process
from static.inference import Inference
from flask_cors import CORS, cross_origin



app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# accept a JSON object 
@app.route('/static/cnn', methods=['POST'])
def index():
    data = request.get_json()
    image = Process(data['imageSrc']).readb64()
    inference = Inference(image).preProcess()
    print(inference)
    return ""

if __name__ == '__main__':
    app.run(debug=True)