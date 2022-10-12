import sys
import os
import time
from typing import Optional
import subprocess

# instance group names
DATA_GROUP = 'data_group' 
DNN_GROUP = 'dnn_group'


def start_child_process_async(name : str, additional_args=[]) -> int:
    #TODO: Find a way to stream stdout and stderr to the parent process
    params = ["python", f"./{name}"] + sys.argv[1:] + additional_args
    print(f'Opening process async: {params}')
    p = subprocess.Popen(params)
    print(f'Process {name} started')
    return p.pid


def start_child_process(name : str, additional_args=[]) -> int:
     params = ["python", f"./{name}"] + sys.argv[1:] + additional_args
     print(f'Opening process: {params}')
     p = subprocess.run(params)
     print(f'Process {name} closed with returncode={p.returncode}')
     return p.returncode


def start_data_group(dispatcher_host : str) -> int:
    return start_child_process('train_data.py', ["--dispatcher_host", dispatcher_host])


def not_mpi_or_rank_0() -> bool:
    return 'OMPI_COMM_WORLD_LOCAL_RANK' not in os.environ or os.environ['OMPI_COMM_WORLD_LOCAL_RANK'] == '0'


def start_dnn_group(dispatcher_host : Optional[str]) -> int:
    if dispatcher_host is not None:
        args = ["--dispatcher_host", dispatcher_host]
        # Start a tf.data.service worker processes for each host in the DNN group
        # to take advantage of its CPU resources. 
        # Start once per instance, not per MPI process
        if not_mpi_or_rank_0(): 
            start_child_process_async('train_data.py', args)
    else:
        args = []
    return start_child_process('train_dnn.py', args)


def get_group_first_host(instance_groups, target_group_name):
        return instance_groups[target_group_name]['hosts'][0]


def is_not_mpi_or_world_rank_0() -> bool:
    return 'OMPI_COMM_WORLD_RANK' in os.environ and os.environ['OMPI_COMM_WORLD_RANK'] != '0'


def shutdown_tf_data_service_with_retries(hosts : list):
    # only world rank 0 process should shutdown the dispatcher
    if is_not_mpi_or_world_rank_0():
        return 

    completed_hosts = []
    for host in hosts:
        for i in range(0,12):
            try:
                if i>0:
                    sleeptime = 10
                    print(f'Will attempt {i} time to shutdown in {sleeptime} seconds')
                    time.sleep(sleeptime)
                
                if host not in completed_hosts:
                    _shutdown_data_service(host)
                    completed_hosts.append(host)
                break
            except Exception as e:
                print(f'Failed to shutdown dispatcher in {host} due to: {e}')


def _shutdown_data_service(dispatcher_host : str):
    SHUTDOWN_PORT = 16000
    print(f'Shutting down tf.data.service dispatcher via: [{dispatcher_host}:{SHUTDOWN_PORT}]')
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((dispatcher_host, SHUTDOWN_PORT))
        print(f'Shutdown request sent to {dispatcher_host}:{SHUTDOWN_PORT}')


def split_to_instance_group_train_script() -> int:
    from sagemaker_training import environment
    env = environment.Environment()
    
    print(f'env.is_hetero={env.is_hetero}')
    print(f'current_host={env.current_host}')
    
    if env.is_hetero:
        dispatcher_host = get_group_first_host(env.instance_groups_dict, DATA_GROUP)
        first_host_in_dnn_group = get_group_first_host(env.instance_groups_dict, DNN_GROUP)
        print(f'current_instance_type={env.current_instance_type}')
        print(f'current_group_name={env.current_instance_group}')
        print(f'dispatcher_host={dispatcher_host}')
        if env.current_instance_group == DATA_GROUP:
            return start_data_group(dispatcher_host)
        elif env.current_instance_group == DNN_GROUP:
            returncode = start_dnn_group(dispatcher_host)
            # first host in DNN group will take care of shutting down the dispatcher
            if env.current_host == first_host_in_dnn_group:
                hosts = env.instance_groups_dict[DATA_GROUP]['hosts'] + env.instance_groups_dict[DNN_GROUP]['hosts']
                shutdown_tf_data_service_with_retries(hosts)
            return returncode
        else:
            raise Exception(f'Unknown instance group: {env.current_instance_group}')
            
    else: # not heterogenous 
        return start_dnn_group(dispatcher_host=None)

if __name__ == "__main__":
    try:
        returncode = split_to_instance_group_train_script()
        exit(returncode)
    except Exception as e:
        print(f'Failed due to {e}. exiting with returncode=1')
        sys.exit(1)