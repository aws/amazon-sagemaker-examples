#!/bin/sh

endpoint_name=$1
api_url=$2

env endpoint=${endpoint_name} locust -f locust_file.py --headless --csv="test_${endpoint_name}" -u 4000 -r 2 --host ${api_url}
