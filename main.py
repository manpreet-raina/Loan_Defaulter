import numpy as np
import pandas as pd
from pre_processing_pipeline import PreProcess as pp1
from ml_model import preprocessed_dataframe
from ml_model import load_model_from_db
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier


def load_model(X, model):
    ml_model = load_model_from_db(model)
    y_pred = ml_model.predict(X)
    for i in y_pred:
        print(i)


if __name__ == '__main__':

    obj1 = pp1()
    df = obj1.pre_processing()
    preprocessed_dataframe()
