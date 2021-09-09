#!/usr/bin/env python

import os
import pandas as pd
from glob import glob
from tqdm import tqdm

os.system("du -a /opt/ml")

SRCTRAINFILE = glob("/opt/ml/processing/input_train/*.csv")[0]
print(SRCTRAINFILE)
SRCVALIDATIONFILE = glob("/opt/ml/processing/input_test/*.csv")[0]
print(SRCVALIDATIONFILE)

DSTTRAINFILE = "/opt/ml/processing/train/train.csv"
DSTVALIDATIONFILE = "/opt/ml/processing/test/test.csv"

# Preparation of the train set
trainFrame = pd.read_csv(SRCTRAINFILE, header=None)
tqdm.pandas()

# Amazon Comprehend "recommend[s] that you train the model with up to 1,000 training documents for
# each label". and no more than 1000000 documents.
#
# Here, we are limiting to a total of 1000 documents in order to reduce costs of this demo.
#
# If you want to test Amazon Comprehend on the full dataset, set MAXITEM to 100000
# MAXITEM=100000

MAXITEM = 1000
# Keeping MAXITEM for each label
for i in range(1, 11):
    num = len(trainFrame[trainFrame[0] == i])
    dropnum = num - MAXITEM
    indextodrop = trainFrame[trainFrame[0] == i].sample(n=dropnum).index
    trainFrame.drop(indextodrop, inplace=True)

# Applying translation of numerical codes into labels
trainFrame[0] = trainFrame[0].progress_apply(
    {
        1: "SOCIETY_AND_CULTURE",
        2: "SCIENCE_AND_MATHEMATICS",
        3: "HEALTH",
        4: "EDUCATION_AND_REFERENCE",
        5: "COMPUTERS_AND_INTERNET",
        6: "SPORTS",
        7: "BUSINESS_AND_FINANCE",
        8: "ENTERTAINMENT_AND_MUSIC",
        9: "FAMILY_AND_RELATIONSHIPS",
        10: "POLITICS_AND_GOVERNMENT",
    }.get
)
# Joining "Question title", "question content", and "best answer".
trainFrame["document"] = trainFrame[trainFrame.columns[1:]].progress_apply(
    lambda x: " \\n ".join(x.dropna().astype(str)), axis=1
)
# Keeping only the first two columns: label and joint text
trainFrame.drop([1, 2, 3], axis=1, inplace=True)
# Escaping ','
trainFrame["document"] = trainFrame["document"].str.replace(",", "&#44;")
# Writing csv file
trainFrame.to_csv(
    path_or_buf=DSTTRAINFILE,
    header=False,
    index=False,
    escapechar="\\",
    doublequote=False,
    quotechar='"',
)


# Preparation of the validation set
validationFrame = pd.read_csv(SRCVALIDATIONFILE, header=None)
tqdm.pandas()

# Here, we are limiting to 1000 documents to test in order to reduce costs of this demo.
# If you want to test Amazon Comprehend on the full dataset, set MAXITEM to None
# MAXITEM=None
MAXITEM = 1000
# Keeping MAXITEM
if MAXITEM:
    num = len(validationFrame)
    dropnum = num - MAXITEM
    indextodrop = validationFrame.sample(n=dropnum).index
    validationFrame.drop(indextodrop, inplace=True)

# Joining "Question title", "question content", and "best answer".
validationFrame["document"] = validationFrame[validationFrame.columns[1:]].progress_apply(
    lambda x: " \\n ".join(x.dropna().astype(str)), axis=1
)
# Removing all column but the aggregated one
validationFrame.drop([0, 1, 2, 3], axis=1, inplace=True)
# Escaping ','
validationFrame["document"] = validationFrame["document"].str.replace(",", "&#44;")
# Writing csv file
validationFrame.to_csv(
    path_or_buf=DSTVALIDATIONFILE,
    header=False,
    index=False,
    escapechar="\\",
    doublequote=False,
    quotechar='"',
)
