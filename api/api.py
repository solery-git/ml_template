# -*- coding: utf-8 -*-
import os

from flask import Flask
from flask_restful import Resource, Api, reqparse
from joblib import load
import numpy as np
import pandas as pd

MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_FILE = os.environ["MODEL_FILE"]
METADATA_FILE = os.environ["METADATA_FILE"]
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
METADATA_PATH = os.path.join(MODEL_DIR, METADATA_FILE)

print("Loading model from: {}".format(MODEL_PATH))
model = load(MODEL_PATH)

app = Flask(__name__)
api = Api(app)

class Prediction(Resource):
    def __init__(self):
        self._required_features = ['SEP_LEN', 'SEP_WDTH', 'PET_LEN', 'PET_WDTH']
        self._df_columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        self._reqparser = reqparse.RequestParser()
        for feature in self._required_features:
            self._reqparser.add_argument(
                feature, type = float, required = True, location = 'json',
                help = 'No {} provided'.format(feature))
        super(Prediction, self).__init__()

    def post(self):
        args = self._reqparser.parse_args()
        X = pd.DataFrame(np.array([args[f] for f in self._required_features]).reshape(1, -1), 
                         columns=self._df_columns)
        y_pred = model.predict(X)
        return {'prediction': y_pred.tolist()[0]}

api.add_resource(Prediction, '/predict')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')