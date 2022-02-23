from flask import Flask, request
from static.process import *

app = Flask(__name__)

# accept a JSON object 
@app.route('/static/cnn')
def index():
    return "Hello, World!"

if __name__ == '__main__':
    app.run(debug=True)