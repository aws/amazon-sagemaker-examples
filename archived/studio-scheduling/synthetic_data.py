"""
Generate Synthetic Customer Churn Dataset
"""
from random import randint
from random import choice

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification


def random_with_n_digits(n):
    range_start = 10 ** (n - 1)
    range_end = (10**n) - 1
    return randint(range_start, range_end)


def generate_data(num_samples=5000, label_known=True):
    x, y = make_classification(
        n_samples=num_samples,
        n_features=14,
        n_informative=10,
        n_redundant=3,
        n_classes=2,
        flip_y=0.01,
        shift=5,
        n_clusters_per_class=1,
    )

    df_features = pd.DataFrame(x, columns=FEATURE_COLUMNS)
    df_features = df_features.abs()

    df_features = pd.concat([df_features, pd.Series(y)], axis=1)
    df_features = df_features.rename(columns={0: "Churn?"})

    # Format Columns
    df_features["Churn?"] = df_features["Churn?"].replace(0, "False.")
    df_features["Churn?"] = df_features["Churn?"].replace(1, "True.")

    df_features["Day Calls"] = df_features["Day Calls"].astype(int).abs()
    df_features["Eve Calls"] = df_features["Eve Calls"].astype(int).abs()
    df_features["Night Calls"] = df_features["Night Calls"].astype(int).abs()
    df_features["Intl Calls"] = df_features["Intl Calls"].astype(int).abs()
    df_features["CustServ Calls"] = df_features["CustServ Calls"].astype(int).abs()
    df_features["VMail Message"] = df_features["VMail Message"].astype(int).abs()
    df_features["Night Calls"] = df_features["Night Calls"].astype(int).abs()

    # Scale Columns
    df_features["VMail Message"] = df_features["VMail Message"] * 100
    df_features["Night Calls"] = df_features["Night Calls"] * 50

    # Generate Metadata columns
    state_series = np.random.choice(STATES, num_samples, replace=True)
    df_other = pd.DataFrame(state_series, columns=["State"])
    df_other["Account Length"] = [randint(1, 200) for x in range(num_samples)]
    df_other["Area Code"] = df_other["State"].apply(
        lambda x: (str(ord(x[0])) + str(ord(x[1])))[:3]
    )
    df_other["Phone"] = [
        "{}-{}".format(random_with_n_digits(3), random_with_n_digits(4))
        for x in range(num_samples)
    ]
    df_other["Int'l Plan"] = [choice(["yes", "no"]) for x in range(num_samples)]
    df_other["VMail Plan"] = [choice(["yes", "no"]) for x in range(num_samples)]

    # Consolidate
    df_final = pd.concat([df_other, df_features], axis=1)
    df_final["VMail Message"] = df_final["VMail Message"].mask(
        df_final["VMail Plan"] == "no", 0
    )
    
    if not label_known:
        df_final = df_final.drop(["Churn?"], axis=1)

    return df_final


FEATURE_COLUMNS = [
    "VMail Message",
    "Day Mins",
    "Day Calls",
    "Day Charge",
    "Eve Mins",
    "Eve Calls",
    "Eve Charge",
    "Night Mins",
    "Night Calls",
    "Night Charge",
    "Intl Mins",
    "Intl Calls",
    "Intl Charge",
    "CustServ Calls",
]


STATES = [
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DC",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
]
