from io import StringIO
import pandas as pd
import streamlit as st
import time
import pymongo
from pymongo import MongoClient
import seaborn as sns

from ml_model import load_model_from_db
from pre_processing_pipeline import PreProcess

import numpy as np
import plotly.express as pt


def load_model(X, model):
    #X = np.array(X).astype(np.float32)
    #X.to_csv("/Users/manpreetsingh/Loan Defaulter Prediction/ADT/test.csv")
    xgb = load_model_from_db(model)
    y_pred = xgb.predict(X)
    X["outcome"] = y_pred
    return X


def database():
    # Connection setup
    client = MongoClient("localhost", 27017)
    db = client["LendingClub"]
    return db


def bulk_insert():
    st.markdown("# Bulk Insert")
    st.sidebar.markdown("# Bulk Insert")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        st.write(bytes_data)

        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        st.write(stringio)

        # To read file as string:
        string_data = stringio.read()
        st.write(string_data)

        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file)
        #st.write(dataframe)

        db = database()
        db.staging.insert_many(dataframe.to_dict('records'))
        obj = PreProcess()
        df = obj.pre_process_pipeline()
        new_df1 = load_model(df, "xgb")
        #new_df2 = load_model(df, "rf")
        st.dataframe(new_df1["outcome"])
        #st.dataframe(new_df2)


page_names_to_funcs = {
    "Bulk Insert": bulk_insert,
    # "Dashboard": dashboard,
    # "Page 3": page3,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
