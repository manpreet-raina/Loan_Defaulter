import numpy as np
import pandas as pd
from pre_processing_pipeline import PreProcess as pp1
from ml_model import preprocessed_dataframe
from ml_model import load_model_from_db
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier


def load_model(X, model):
    #scaler_func = MinMaxScaler()

    X = pd.read_csv("/Users/manpreetsingh/Loan Defaulter Prediction/ADT/read/X_test.csv")
    y = pd.read_csv("/Users/manpreetsingh/Loan Defaulter Prediction/ADT/read/y_test.csv")
    #X = scaler_func.transform(X)
    print(X.columns)
    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.float32)

    ml_model = load_model_from_db(model)
    y_pred = ml_model.predict(X)
    for i in y_pred:
        print(i)


if __name__ == '__main__':

    obj1 = pp1()
    df = obj1.pre_processing()
    preprocessed_dataframe()
    #load_model(df, "xgb")
    #load_model(df, "rf")
