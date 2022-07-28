import json
from io import StringIO
import pandas as pd
import streamlit as st
import time
import pymongo
from pymongo import MongoClient
import seaborn as sns

from ml_model import load_model_from_db
from pre_processing_pipeline import PreProcess
from streamlit_echarts import st_echarts

import numpy as np
import plotly.express as pt


def load_model(X, model):
    xgb = load_model_from_db(model)
    y_pred = xgb.predict(X)
    return list(y_pred)


def database():
    # Connection setup
    client = MongoClient("localhost", 27017)
    db = client["LendingClub"]
    return db


def encoding(row):
    if row.outcome == 1:
        return "Approved"
    else:
        return "Denied"


def bulk_insert():
    st.markdown("# Bulk Insert")
    st.sidebar.markdown("# Bulk Insert")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)

        db = database()
        db.staging.insert_many(dataframe.to_dict('records'))
        obj = PreProcess()
        df = obj.pre_process_pipeline()
        y1 = load_model(df, "xgb")
        dataframe["outcome"] = y1
        dataframe["Application"] = dataframe.apply(lambda x: encoding(x), axis=1)
        st.write(dataframe)


def loan_status_distribution():
    db = database()
    cursor = db.loan_data.find({}, {"loan_status": 1, "_id": 0})
    df1 = pd.DataFrame(list(cursor))
    df1.dropna(inplace=True)
    df1_dict = dict(df1['loan_status'].value_counts())

    val1 = df1_dict["Fully Paid"]
    val2 = df1_dict["Charged Off"]
    options = {
        "tooltip": {"trigger": "item"},
        "legend": {"top": "5%", "left": "center"},
        "series": [
            {
                "name": "Loan Status",
                "type": "pie",
                "radius": ["40%", "70%"],
                "avoidLabelOverlap": False,
                "itemStyle": {
                    "borderRadius": 10,
                    "borderColor": "#fff",
                    "borderWidth": 2,
                },
                "label": {"show": False, "position": "center"},
                "emphasis": {
                    "label": {"show": True, "fontSize": "40", "fontWeight": "bold"}
                },
                "labelLine": {"show": False},
                "data": [
                    {"value": int(val1), "name": "Fully Paid"},
                    {"value": int(val2), "name": "Charged Off"},
                ],
            }
        ],
    }
    st_echarts(
        options=options, height="500px",
    )


page_names_to_funcs = {
    "Bulk Insert": bulk_insert,
    "Loan Status Distribution": loan_status_distribution,
    "loan status distribution against numerical  features": loan_status_distribution_against_numerical_features,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
