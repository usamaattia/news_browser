from flask import Flask, render_template, request, jsonify
import pandas as pd
#from bm25 import bm



app = Flask(__name__)



@app.route("/")
def home():
    return render_template("index.html")


@app.route("/results")
def res():
    return render_template("results.html")


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)