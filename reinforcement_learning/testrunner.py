#!/usr/bin/env python
# coding: utf-8

import itertools
import json
import os
import subprocess
import sys
import time
try:
    import pandas as pd
    import papermill
    from tabulate import tabulate
except ImportError:
    sys.exit("""Some libraries are missing. Please install them by running `pip install -r test_requirements.txt`.""")

# CONSTANTS
TEST_NOTEBOOKS_FILE = 'testnotebooks.txt'
TEST_CONFIG_FILE = 'testconfig.csv'
SUCCESSES = 0
EXCEPTIONS = 0
SUCCESSFUL_EXECUTIONS = []
FAILED_EXECUTIONS = []
CELL_EXECUTION_TIMEOUT_SECONDS = 1200


# helper functions
def run_notebook(nb_path, test_config):
    dir_name = os.path.dirname(nb_path)
    nb_name = os.path.basename(nb_path)
    output_nb_name = "output_{}.ipynb".format(nb_name)
    os.chdir(dir_name)
    print("Current directory: {}".format(os.getcwd()))
    global SUCCESSES
    global EXCEPTIONS
    for i in range(len(test_config)):
        params = json.loads(test_config.loc[i].to_json())
        # Coach notebooks support only single instance training, so skip the tests with multiple EC2 instances
        if 'coach' in nb_name.lower() and params['train_instance_count'] > 1:
            continue
        print("\nTEST: " + nb_name + " with parameters " + str(params))
        process = None
        try:
            papermill.execute_notebook(nb_name, output_nb_name, parameters=params, execution_timeout=CELL_EXECUTION_TIMEOUT_SECONDS, log_output=True)
            SUCCESSES += 1
            SUCCESSFUL_EXECUTIONS.append(dict({'notebook':nb_name, 'params':params}))
        except BaseException as error:
            print('An exception occurred: {}'.format(error))
            EXCEPTIONS += 1
            FAILED_EXECUTIONS.append(dict({'notebook':nb_name, 'params':params}))

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



notebooks_list = open(TEST_NOTEBOOKS_FILE).readlines()
config = pd.read_csv(TEST_CONFIG_FILE)
ROOT = os.path.abspath('.')

# Run tests on each notebook listed in the config.
print("Test Configuration: ")
print(config)
for nb_path in notebooks_list:
    os.chdir(ROOT)
    print("Testing: {}".format(nb_path))
    run_notebook(nb_path.strip(), config)

# Print summary of tests ran.
print("Summary: {}/{} tests passed.".format(SUCCESSES, SUCCESSES + EXCEPTIONS))
print("Successful executions: ")
print_notebook_executions(SUCCESSFUL_EXECUTIONS)

# Throw exception if any test fails, so that the CodeBuild also fails.
if EXCEPTIONS > 0:
    print("Failed executions: ")
    print_notebook_executions(FAILED_EXECUTIONS)
    raise Exception("Test did not complete successfully")
