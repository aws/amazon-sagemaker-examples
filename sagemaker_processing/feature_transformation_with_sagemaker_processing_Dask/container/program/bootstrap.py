#!/usr/bin/env python3
import json
import os
import socket
import subprocess
import sys
import time
from shutil import copyfile
from subprocess import PIPE, Popen

DASK_PATH = "/opt/conda/bin"


def get_resource_config():
    resource_config_path = "/opt/ml/config/resourceconfig.json"
    with open(resource_config_path, "r") as f:
        return json.load(f)


def start_daemons(master_ip):
    resource_config = get_resource_config()
    current_host = resource_config["current_host"]
    scheduler_host = resource_config["hosts"][0]

    cmd_start_scheduler = os.path.join(DASK_PATH, "dask-scheduler")
    cmd_start_worker = os.path.join(DASK_PATH, "dask-worker")
    schedule_conn_string = "tcp://{ip}:8786".format(ip=master_ip)
    if current_host == scheduler_host:
        Popen([cmd_start_scheduler])
        Popen([cmd_start_worker, schedule_conn_string])
    else:
        worker_process = Popen([cmd_start_worker, schedule_conn_string])


def get_ip_from_host(host_name):
    ip_wait_time = 200
    counter = 0
    ip = ""

    while counter < ip_wait_time and ip == "":
        try:
            ip = socket.gethostbyname(host_name)
            break
        except:
            counter += 1
            time.sleep(1)

    if counter == ip_wait_time and ip == "":
        raise Exception(
            "Exceeded max wait time of {}s for hostname resolution".format(ip_wait_time)
        )

    return ip


if __name__ == "__main__":
    ips = []
    resource_config = get_resource_config()
    master_host = resource_config["hosts"][0]
    scheduler_ip = get_ip_from_host(master_host)
    current_host = resource_config["current_host"]

    # Start Dask cluster in all nodes
    start_daemons(scheduler_ip)

    # wait for a few seconds until the cluster is up and running
    time.sleep(10)

    # Submit the preprocessing job on the cluster from the first instance. You only need to submit the job once from one node.
    if current_host == master_host:
        cmd_string = ["/opt/conda/bin/python", str(sys.argv[1])]
        cmd_string.extend(sys.argv[2:])
        cmd_string.append(scheduler_ip)
        result = subprocess.Popen(cmd_string)
        _ = result.communicate()[0]
        exit_code = result.returncode
    else:
        while True:
            scheduler = (scheduler_ip, 8786)
            alive_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            alive_check = alive_socket.connect_ex(scheduler)
            if alive_check == 0:
                pass
            else:
                print("Received a shutdown signal from Dask cluster")
                sys.exit(0)
            alive_socket.close()
            time.sleep(2)
