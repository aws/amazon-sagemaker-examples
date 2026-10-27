import sys
import time
from typing import Optional

# instance group names
DATA_GROUP = 'data_group' 
DNN_GROUP = 'dnn_group'

def start_child_process(name : str, additional_args=[]) -> int:
     import subprocess
     params = ["python", f"./{name}"] + sys.argv[1:] + additional_args
     print(f'Opening process: {params}')
     p = subprocess.run(params)
     print(f'Process {name} closed with returncode={p.returncode}')
     if p.returncode == -15 or p.returncode == -9:
        print(f'Received SIGTERM|SIGKILL which is normal termination for pytorch data service to avoid hanging process')
        return 0
     return p.returncode


def start_data_group(dispatcher_host : str) -> int:
    return start_child_process('train_data.py', ["--dispatcher_host", dispatcher_host])


def start_dnn_group(dispatcher_host : Optional[str]) -> int:
    additional_args = [] if dispatcher_host is None else ["--dispatcher_host", dispatcher_host]
    return start_child_process('train_dnn.py', additional_args)


def get_group_first_host(instance_groups, target_group_name):
        return instance_groups[target_group_name]['hosts'][0]

def shutdown_pt_data_service_with_retries(dispatcher_host : str):
    for i in range(0,12):
        try:
            if i>0:
                sleeptime = 10
                print(f'Will attempt {i} time to shutdown in {sleeptime} seconds')
                time.sleep(sleeptime)
            _shutdown_data_service(dispatcher_host)
            break
        except Exception as e:
            print(f'Failed to shutdown dispatcher in {dispatcher_host} due to: {e}')


def _shutdown_data_service(dispatcher_host : str):
    SHUTDOWN_PORT = 16000
    print(f'Shutting down data service dispatcher via: [{dispatcher_host}:{SHUTDOWN_PORT}]')
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((dispatcher_host, SHUTDOWN_PORT))
        print(f'Shutdown request sent to {dispatcher_host}:{SHUTDOWN_PORT}')


def split_to_instance_group_train_script() -> int:
    from sagemaker_training import environment
    env = environment.Environment()
    # try:
    #     from sagemaker_training import environment
    #     env = environment.Environment()
    # except ImportError:
    #     class Object(object):
    #         pass

    #     env = Object()
    #     env.is_hetero = True
    #     env.current_host = 'dummyhost'
    #     env.instance_groups_dict = {DATA_GROUP : {'hosts': ['dummyhost']}}
    #     env.current_instance_group = DNN_GROUP
    #     env.current_instance_type = 'dummyinstance'
    
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
            # first host in DNN group takes care of shutting down the dispatcher
            if env.current_host == first_host_in_dnn_group:
                shutdown_pt_data_service_with_retries(dispatcher_host)
            return returncode
        else:
            raise Exception(f'Unknown instance group: {env.current_instance_group}')
            
    else: # not hetero 
        return start_dnn_group(dispatcher_host=None)

if __name__ == "__main__":
    try:
        returncode = split_to_instance_group_train_script()
        exit(returncode)
    except Exception as e:
        print(f'Failed due to {e}. exiting with returncode=1')
        sys.exit(1)