import streamlit as st
import pandas as pd
import os
import ydata_profiling
from ydata_profiling import profile_report, ProfileReport
from streamlit_pandas_profiling import st_profile_report
import pycaret.classification as pcl
import pycaret.regression as pr


df = None


if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

with st.sidebar:
    st.image("images/ML icon.png")
    st.title("AutoML")
    choice = st.radio("Navigation", ["Upload", "Profiling", "ML", "Download"])
    st.info("This application allows you to build automated ML pipeline using streamlit, "
            "Pandas profiling and PyCaret.And it works like magic!")


if choice == "Upload":
    st.title("Upload Your Data For Modelling!")
    file = st.file_uploader("Upload your Dataset here")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=False)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Exploratory Data Analysis")
    if st.button("Generate report"):
        report = df.profile_report()
        st_profile_report(report)


elif choice == "ML":
    st.title("Machine Learning Model Building!")
    target = st.selectbox("Select Your Target", df.columns)
    model_type = st.selectbox("Select mode type", ['Regression', 'Classification'])
    if model_type == "Classification":
        if st.button("Train model"):
            pcl.setup(df, target=target)
            setup_df = pcl.pull()
            st.info("This is ML Experiment settings")
            st.dataframe(setup_df)
            best_model = pcl.compare_models()
            compare_df = pcl.pull()
            st.info("This is the ML Model")
            st.dataframe(compare_df)
            best_model
            pcl.save_model(best_model, 'best_model')

    elif model_type == "Regression":
        if st.button("Train model"):
            pr.setup(df, target=target)
            setup_df = pr.pull()
            st.info("This is ML Experiment settings")
            st.dataframe(setup_df)
            best_model = pr.compare_models()
            compare_df = pr.pull()
            st.info("This is the ML Model")
            st.dataframe(compare_df)
            best_model
            pr.save_model(best_model, 'best_model')


elif choice == "Download":
    with open("best_model.pkl", 'rb') as f:
        st.download_button("Download the Model", f, "trained_model.pkl")


