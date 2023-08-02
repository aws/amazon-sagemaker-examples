import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st

st.title("Breast Cancer Analysis")
data_url = "https://sagemaker-sample-files.s3.amazonaws.com/datasets/tabular/breast_cancer/breast-cancer-wisconsin.csv"
columns = [
    "Id",
    "Cl.thickness",
    "Cell.size",
    "Cell.shape",
    "Marg.adhesion",
    "Epith.c.size",
    "Bare.nuclei",
    "Bl.cromatin",
    "Normal.nucleoli",
    "Mitoses",
    "Class",
]


@st.cache
def load_data():
    df = pd.read_csv(data_url, names=columns)
    df["Class"] = df["Class"].replace(to_replace=[2, 4], value=["benign", "malignant"])
    return df


data_load_state = st.text("Loading data...")
data = load_data()
data_load_state.text("")

column = st.selectbox("Features", columns[1:-1])

st.subheader("Histogram of %s by diagnosis type" % column)
fig, ax = plt.subplots(
    1,
    2,
    sharex=True,
    sharey=True,
)
data.hist(column=column, by=columns[-1], layout=(1, 2), ax=ax)
st.pyplot(fig)

if st.checkbox("Show raw data"):
    st.subheader("Raw data")
    st.write(data[[column, columns[-1]]])

st.markdown(f"Source: <{data_url}>")
