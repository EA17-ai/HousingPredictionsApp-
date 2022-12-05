import pickle
import pandas as pd
from pycaret.regression import load_model, predict_model
import numpy as np
import requests
from flask import Flask, render_template, request
import json
import sklearn

app = Flask(__name__)
columns_df = json.load(open("columns.json"))
total_col=columns_df["data_columns"]
global column
column = columns_df["data_columns"][3:]

global pretrained_model
with open("housingmodel_correct.pkl", "rb") as f:
    pretrained_model = pickle.load(f)


def get_predictions(total_sqft, bath, bhk, location):
    try:
        loc_index = total_col.index(location.lower())
    except:
        loc_index = -1
    x = np.zeros(len(total_col))
    x[0] = total_sqft
    x[1] = bath
    x[2] = bhk
    if loc_index > 0:
        x[loc_index] = 1

    return round(pretrained_model.predict([x])[0],2)


@app.route("/")
def home():
    return render_template("home.html", column=column)


@app.route("/page", methods=['GET', 'POST'])
def predict():
    location = request.form.get("location")
    bhk = request.form.get("bhk")
    bath = request.form.get("bath")
    sqft = request.form.get("sqft")
    price = get_predictions(total_sqft=sqft, bath=bath, bhk=bhk, location=location)
    print(location)
    return render_template("page.html", price=price, total_sqft=sqft, bath=bath, bhk=bhk, location=location)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(debug=True)
