import numpy as np
import pickle
import time

import pymongo
from pymongo import MongoClient

from spark_connection import SparkFunctions
from config import mongo_ip
from pre_processing_pipeline import PreProcess
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
)

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


def model_to_db(model, name):
    pickle_model = pickle.dumps(model)
    client = MongoClient("localhost", 27017)
    db = client["LendingClub"]
    db.model.insert_one({"model": pickle_model, "name": name, "added_timestamp": time.time()})


def load_model_from_db(name):
    json_data = {}
    client = MongoClient("localhost", 27017)
    db = client["LendingClub"]
    data = db.model.find({"name": name})

    for i in data:
        print(i)
        json_data = i

    pickled_model = json_data["model"]

    return pickle.loads(pickled_model)


def print_score(true, pred, train=True):
    if train:
        clf_report = pd.DataFrame(classification_report(true, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(true, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(true, pred)}\n")

    elif train == False:
        clf_report = pd.DataFrame(classification_report(true, pred, output_dict=True))
        print("Test Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(true, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(true, pred)}\n")


def encoding(row):
    if row.loan_status == "Fully Paid":
        return 1
    else:
        return 0


def preprocessed_dataframe():
    sf = SparkFunctions
    sc, sql_c = sf.spark_context(sf)
    preprocessed_df = sql_c.read.format("com.mongodb.spark.sql.DefaultSource"). \
        option("uri", mongo_ip + "pre_processed_data").load()
    train, test = preprocessed_df.randomSplit([0.67, 0.33], seed=42)
    train = train[train['annual_inc'] <= 250000]
    train = train[train['dti'] <= 50]
    train = train[train['open_acc'] <= 40]
    train = train[train['total_acc'] <= 80]
    train = train[train['revol_util'] <= 120]
    train = train[train['revol_bal'] <= 250000]
    train = train.drop('_id')
    train = train.drop('added_timestamp')
    test = test.drop('_id')
    test = test.drop('added_timestamp')
    test = test.toPandas()
    train = train.toPandas()
    y_train = train[["loan_status"]]
    y_test = test[["loan_status"]]
    X_train = train.drop('loan_status', axis=1)
    X_test = test.drop('loan_status', axis=1)
    y_train.loan_status = y_train.apply(lambda x: encoding(x), axis=1)
    y_test.loan_status = y_test.apply(lambda x: encoding(x), axis=1)

    X_test.to_csv("/Users/manpreetsingh/Loan Defaulter Prediction/ADT/X_test.csv", index=False)
    y_test.to_csv("/Users/manpreetsingh/Loan Defaulter Prediction/ADT/y_test.csv", index=False)
    scaler_func = MinMaxScaler()
    X_train = scaler_func.fit_transform(X_train)
    X_test = scaler_func.transform(X_test)
    X_train = np.array(X_train).astype(np.float32)
    X_test = np.array(X_test).astype(np.float32)
    y_train = np.array(y_train).astype(np.float32)
    y_test = np.array(y_test).astype(np.float32)
    xgb_model(X_train, y_train, X_test, y_test)
    #rf_model(X_train, y_train, X_test, y_test)


def xgb_model(X_train, y_train, X_test, y_test):
    # xgb classifier
    xgb = XGBClassifier(use_label_encoder=False)
    # If you want to test model
    """
    xgb.fit(X_train, y_train)
    y_train_pred = xgb.predict(X_train)
    y_test_pred = xgb.predict(X_test)
    print_score(y_train, y_train_pred, train=True)
    print_score(y_test, y_test_pred, train=False)
    """

    xgb.fit(X_train, y_train)
    model_to_db(xgb, "xgb")
    xgb_test = load_model_from_db("xgb")
    y_train_pred = xgb_test.predict(X_train)
    y_test_pred = xgb_test.predict(X_test)
    print_score(y_train, y_train_pred, train=True)
    print_score(y_test, y_test_pred, train=False)


"""if __name__ == '__main__':
    obj = PreProcess()
    obj.pre_processing()
    preprocessed_dataframe()"""
