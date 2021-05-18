import datetime
import os

START_DATE = "2012-08-13"
END_DATE = "2017-08-11"
DATE_FORMAT = "%Y-%m-%d"
EPS = 1e-8

CUR_DIR = os.path.dirname(__file__)
DATA_DIR = CUR_DIR + "/datasets/stocks_history_target.h5"
CSV_DIR = "/opt/ml/output/data/portfolio-management.csv"
START_DATETIME = datetime.datetime.strptime(START_DATE, DATE_FORMAT)
