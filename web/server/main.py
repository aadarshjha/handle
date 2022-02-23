from flask import Flask

app = Flask(__name__)

@app.route("/static/cnn")
def cnn(): 
    return "Hello CNN"