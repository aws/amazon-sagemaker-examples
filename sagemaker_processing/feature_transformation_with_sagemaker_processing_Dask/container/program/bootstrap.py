#!/usr/bin/env python3
import os
import socket
import json
import psutil
import subprocess
import sys
import time
from dask.distributed import Scheduler, Worker, Client
from shutil import copyfile
from subprocess import Popen, PIPE



DASK_PATH = '/opt/conda/bin/'


def get_resource_config():
    resource_config_path = '/opt/ml/config/resourceconfig.json'
    with open(resource_config_path, 'r') as f:
        return json.load(f)





def start_daemons(master_ip):
    resource_config = get_resource_config()
    current_host = resource_config['current_host']
    scheduler_host = resource_config['hosts'][0]

    cmd_start_scheduler = DASK_PATH + 'dask-scheduler'
    cmd_start_worker = DASK_PATH + 'dask-worker'

    if current_host == scheduler_host:
        Popen([cmd_start_scheduler])
        Popen([cmd_start_worker , 'tcp://' + str(master_ip) + ':8786'])
    else:
        worker_process = Popen([cmd_start_worker , 'tcp://' + str(master_ip) + ':8786'])



def get_ip_from_host(host_name):
    IP_WAIT_TIME = 100
    counter = 0
    ip = ''

    while counter < IP_WAIT_TIME and ip == '':
        try:
            ip = socket.gethostbyname(host_name)
            break
        except:
            counter += 1
            time.sleep(1)

    if counter == IP_WAIT_TIME and ip == '':
        raise Exception("Exceeded max wait time of 100s for hostname resolution")

    return ip


if __name__ == "__main__":
    ips = []
    resource_config = get_resource_config()
    master_host = resource_config['hosts'][0]
    scheduler_ip = get_ip_from_host(master_host)
    current_host = resource_config['current_host']
    alive_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    start_daemons(scheduler_ip)

   
    time.sleep(20)
    if current_host == master_host:
        cmd_string = ['/opt/conda/bin/python', str(sys.argv[1])]
        cmd_string.extend(sys.argv[2:])
        cmd_string.append(str(scheduler_ip))
        result = subprocess.Popen(cmd_string)
        _ = result.communicate()[0]
        exit_code = result.returncode
    else:
        while True:
            scheduler = (scheduler_ip, 8786)
            alive_check = alive_socket.connect_ex(scheduler)

            if alive_check == 0:
                pass
            else:
                print("Received a shutdown signal from Dask cluster")
                sys.exit(0)
            time.sleep(2)