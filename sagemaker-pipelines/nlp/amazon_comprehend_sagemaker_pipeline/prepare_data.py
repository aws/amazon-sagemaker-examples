#!/usr/bin/env python

import os
import pandas as pd
from glob import glob
from tqdm import tqdm

os.system("du -a /opt/ml")

SRCTRAINFILE = glob("/opt/ml/processing/input_train/*.csv")[0]
print(SRCTRAINFILE)
SRCTESTFILE = glob("/opt/ml/processing/input_test/*.csv")[0]
print(SRCTESTFILE)

DSTTRAINFILE = "/opt/ml/processing/train/train.csv"
DSTTESTFILE = "/opt/ml/processing/test/test.csv"

# Preparation of the train set
trainFrame = pd.read_csv(SRCTRAINFILE, header=None)
testFrame = pd.read_csv(SRCTESTFILE, header=None)

# AWS recommends that you train an Amazon Comprehend model with at least 50 training documents for
# each class. See: https://docs.aws.amazon.com/comprehend/latest/dg/how-document-classification-training-data.html
#
# The dataset we use has 100,000 documents per class. To limit the costs and training times of this demo,
# we will limit it to 1000 documents per class
#
# If you want to test Amazon Comprehend on the full dataset, set MAXITEM to 100000

MAXITEM = 1000
# Keeping MAXITEM for each label
for i in trainFrame[0].unique():
    num = len(trainFrame[trainFrame[0] == i])
    dropnum = max(0, num - MAXITEM)
    indextodrop = trainFrame[trainFrame[0] == i].sample(n=dropnum).index
    trainFrame.drop(indextodrop, inplace=True)

# Escaping commas in preparation to write the data to a CSV file
trainFrame[1] = trainFrame[1].str.replace(",", "&#44;")
# Writing csv file
trainFrame.to_csv(
    path_or_buf=DSTTRAINFILE,
    header=False,
    index=False,
    escapechar="\\",
    doublequote=False,
    quotechar='"',
)

# Escaping commas in preparation to write the data to a CSV file
testFrame[0] = testFrame[0].str.replace(",", "&#44;")
# Writing csv file
testFrame.to_csv(
    path_or_buf=DSTTESTFILE,
    header=False,
    index=False,
    escapechar="\\",
    doublequote=False,
    quotechar='"',
)
