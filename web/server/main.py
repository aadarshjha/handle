from flask import Flask, request
from static.process import *
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# accept a JSON object 
@app.route('/static/cnn', methods=['POST'])
def index():
    data = request.get_json()
    print(data)
    return "Got it"


# @app.route('/static/cnn')
# def index():
#     return "Hello, World!"

if __name__ == '__main__':
    app.run(debug=True)