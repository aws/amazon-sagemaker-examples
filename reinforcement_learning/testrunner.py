#!/usr/bin/env python
# coding: utf-8
from datetime import datetime
startTime = datetime.now()

import itertools
import json
import multiprocessing as mp
import os
import sys
import time
try:
    import pandas as pd
    import papermill
    from tabulate import tabulate
except ImportError:
    sys.exit("""Some libraries are missing. Please install them by running `pip install -r test_requirements.txt`.""")

# CONSTANTS
manager = mp.Manager()
TEST_NOTEBOOKS_FILE = 'testnotebooks.txt'
TEST_CONFIG_FILE = 'testconfig.csv'
SUCCESSES = mp.Value('d', 0)
EXCEPTIONS = mp.Value('d', 0)
SUCCESSFUL_EXECUTIONS = manager.list()
FAILED_EXECUTIONS = manager.list()
CELL_EXECUTION_TIMEOUT_SECONDS = 1200
ROOT = os.path.abspath('.')

jobs = []


# helper functions

def execute_nb_with_params(nb_path, params):
    abs_nb_dir_path = os.path.join(ROOT, os.path.dirname(nb_path))
    nb_name = os.path.basename(nb_path)
    output_nb_name = "output_{}.ipynb".format(nb_name)
    os.chdir(abs_nb_dir_path)
    print("Current directory: {}".format(os.getcwd()))
    print("RUN: " + nb_name + " with parameters " + str(params))
    # Execute notebook
    test_case = dict({'notebook':nb_name, 'params':params})
    try:
        papermill.execute_notebook(nb_name, output_nb_name, parameters=params, execution_timeout=CELL_EXECUTION_TIMEOUT_SECONDS, log_output=True)
        SUCCESSES.value += 1
        SUCCESSFUL_EXECUTIONS.append(test_case)
    except BaseException as error:
        print('An exception occurred: {}'.format(error))
        EXCEPTIONS.value += 1
        FAILED_EXECUTIONS.append(test_case)
    os.chdir(ROOT)


def test_notebook(nb_path, df_test_config):
    for i in range(len(df_test_config)):
        params = json.loads(df_test_config.loc[i].to_json())
        # Coach notebooks support only single instance training, so skip the tests with multiple EC2 instances
        if 'coach' in nb_path.lower() and params['train_instance_count'] > 1:
            continue
        p = mp.Process(target=execute_nb_with_params, args=(nb_path, params))
        time.sleep(1)
        jobs.append(p)
        p.start()



def print_notebook_executions(nb_list_with_params):
    # This expects a list of dict type items.
    # E.g. [{'nb_name':'foo', 'params':'bar'}]
    if not nb_list_with_params:
        print("None")
        return
    vals = []
    for nb_dict in nb_list_with_params:
        val = []
        for k,v in nb_dict.items():
            val.append(v)
        vals.append(val)
    keys = [k for k in nb_list_with_params[0].keys()]
    print(tabulate(pd.DataFrame([v for v in vals], columns=keys), showindex=False))


if __name__ == "__main__":
    notebooks_list = open(TEST_NOTEBOOKS_FILE).readlines()
    config = pd.read_csv(TEST_CONFIG_FILE)
    # Run tests on each notebook listed in the config.
    print("Test Configuration: ")
    print(config)

    for nb_path in notebooks_list:
        print("Testing: {}".format(nb_path))
        test_notebook(nb_path.strip(), config)

    for job in jobs:
        job.join()

    # Print summary of tests ran.
    print("Summary: {}/{} tests passed.".format(SUCCESSES.value, SUCCESSES.value + EXCEPTIONS.value))
    print("Successful executions: ")
    print_notebook_executions(SUCCESSFUL_EXECUTIONS)

    # Throw exception if any test fails, so that the CodeBuild also fails.
    if EXCEPTIONS.value > 0:
        print("Failed executions: ")
        print_notebook_executions(FAILED_EXECUTIONS)
        raise Exception("Test did not complete successfully")

print("Total time taken for tests: ")
print(datetime.now() - startTime)
