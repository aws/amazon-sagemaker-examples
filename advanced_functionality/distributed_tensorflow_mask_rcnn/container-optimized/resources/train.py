import json
import os
import shutil
import subprocess
import sys
import time
import signal
import socket
import glob

from contextlib import contextmanager

def setup():

    # Read info that SageMaker provides
    current_host = os.environ['SM_CURRENT_HOST']
    hosts = json.loads(os.environ['SM_HOSTS'])

    # Enable SSH connections between containers
    _start_ssh_daemon()

    if current_host == _get_master_host_name(hosts):
        _wait_for_worker_nodes_to_start_sshd(hosts)


class TimeoutError(Exception):
    pass

@contextmanager
def timeout(seconds=0, minutes=0, hours=0):
    """
    Add a signal-based timeout to any block of code.
    If multiple time units are specified, they will be added together to determine time limit.
    Usage:
    with timeout(seconds=5):
        my_slow_function(...)
    Args:
        - seconds: The time limit, in seconds.
        - minutes: The time limit, in minutes.
        - hours: The time limit, in hours.
    """

    limit = seconds + 60 * minutes + 3600 * hours

    def handler(signum, frame):  # pylint: disable=W0613
        raise TimeoutError('timed out after {} seconds'.format(limit))

    try:
        signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, limit)
        yield
    finally:
        signal.alarm(0)


def _get_master_host_name(hosts):
    return sorted(hosts)[0]

def _start_ssh_daemon():
    subprocess.Popen(["/usr/sbin/sshd", "-D"])

def _wait_for_worker_nodes_to_start_sshd(hosts, interval=1, timeout_in_seconds=180):
    with timeout(seconds=timeout_in_seconds):
        while hosts:
            print("hosts that aren't SSHable yet: %s", str(hosts))
            for host in hosts:
                ssh_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                if _can_connect(host, 22, ssh_socket):
                    hosts.remove(host)
            time.sleep(interval)


def _can_connect(host, port, s):
    try:
        print("testing connection to host %s", host)
        s.connect((host, port))
        s.close()
        print("can connect to host %s", host)
        return True
    except socket.error:
        print("can't connect to host %s", host)
        return False


def wait_for_training_processes_to_appear_and_finish(proccess_id_string, worker):

    training_process_started = False
    while True:
        time.sleep(300)
        training_process_ps = subprocess.check_output(f'ps -elf | grep "{proccess_id_string}"', encoding='utf-8', shell=True)
        print(training_process_ps)
        training_process_count = subprocess.check_output(f'ps -elf | grep "{proccess_id_string}" | wc -l', encoding='utf-8', shell=True)
        training_process_count_str = training_process_count.replace("\n", "").strip()
        training_process_count = int(training_process_count_str) - 2
        training_process_running = training_process_count > 0
        if training_process_started:
            print(f'training processes running: {training_process_count}')
            if not training_process_running:
                print(f'Worker {worker} training completed.')
                time.sleep(5)
                sys.exit(0)

        if not training_process_started:
            if training_process_running:
                training_process_started = True
            else:
                print(f'Worker {worker} exiting: training not started in 300 seconds.')
                sys.exit(1)

def build_host_arg(host_list, gpu_per_host):
    arg = ""
    for ind, host in enumerate(host_list):
        if ind != 0:
            arg += ","
        arg += f'{host}:{gpu_per_host}'
    return arg

def copy_files(src, dest):
    src_files = os.listdir(src)
    for file in src_files:
        path = os.path.join(src, file)
        if os.path.isfile(path):
            shutil.copy(path, dest)
            
def train():

    import pprint
    pprint.pprint(dict(os.environ), width = 1) 

    model_dir = os.environ['SM_MODEL_DIR']
    log_dir = None
    
    copy_logs_to_model_dir = False
    
    try:
        log_dir = os.environ['SM_CHANNEL_LOG']
        copy_logs_to_model_dir = True
    except KeyError:
        log_dir = model_dir
        
    train_data_dir = os.environ['SM_CHANNEL_TRAIN']
    
    print("pre-setup check")
    setup()

    current_host =  os.environ['SM_CURRENT_HOST']
    all_hosts = json.loads(os.environ['SM_HOSTS'])
    if_name = os.environ['SM_NETWORK_INTERFACE_NAME']

    is_master = current_host == sorted(all_hosts)[0]
    
    if not is_master:
        print(f'Worker: {current_host}')
        process_search_term = "/usr/local/bin/python3.6 /mask-rcnn-tensorflow/MaskRCNN/train.py"
        wait_for_training_processes_to_appear_and_finish(process_search_term, current_host)
        print(f'Worker {current_host} has completed')
    else:
        print(f'Master: {current_host}')
        

    hyperparamters = json.loads(os.environ['SM_HPS'])

    try:
        batch_norm = hyperparamters['batch_norm']
    except KeyError:
        batch_norm = 'FreezeBN'
        
    try:
        mode_fpn = hyperparamters['mode_fpn']
    except KeyError:
        mode_fpn = "True"
        
    try:
        mode_mask = hyperparamters['mode_mask']
    except KeyError:
        mode_mask = "True"

    try:
        eval_period = hyperparamters['eval_period']
    except KeyError:
        eval_period = 1

    try:
        lr_epoch_schedule = hyperparamters['lr_epoch_schedule']
    except KeyError:
        lr_epoch_schedule = '[(16, 0.1), (20, 0.01), (24, None)]'

    try:
        horovod_cycle_time = hyperparamters['horovod_cycle_time']
    except KeyError:
        horovod_cycle_time = 0.5
        
    try:
        horovod_fusion_threshold = hyperparamters['horovod_fusion_threshold']
    except KeyError:
        horovod_fusion_threshold = 67108864

    try:
        data_train = hyperparamters['data_train']
    except KeyError:
        data_train = 'train2017'

    try:
        data_val = hyperparamters['data_val']
    except KeyError:
        data_val = 'val2017'

    try:
        nccl_min_rings = hyperparamters['nccl_min_rings']
    except KeyError:
        nccl_min_rings = 8

    try:
        batch_size_per_gpu = hyperparamters['batch_size_per_gpu']
    except KeyError:
        batch_size_per_gpu = 4
    
    try:
        images_per_epoch = hyperparamters['images_per_epoch']
    except KeyError:
        images_per_epoch = 120000
        
    try:
        backbone_weights = hyperparamters['backbone_weights']
    except KeyError:
        backbone_weights = 'ImageNet-R50-AlignPadding.npz'
    
    try:
        resnet_arch = hyperparamters['resnet_arch']
    except KeyError:
        resnet_arch = 'resnet50'
    
    load_model = None
    try:
        load_model = hyperparamters['load_model']
    except KeyError:
        pass
    
    resnet_num_blocks = '[3, 4, 6, 3]'
    if resnet_arch == 'resnet101':
        resnet_num_blocks = '[3, 4, 23, 3]'
        
    gpus_per_host = int(os.environ['SM_NUM_GPUS'])
    numprocesses = len(all_hosts) * int(gpus_per_host)

    mpirun_cmd = f"""HOROVOD_CYCLE_TIME={horovod_cycle_time} \\
HOROVOD_FUSION_THRESHOLD={horovod_fusion_threshold} \\
mpirun -np {numprocesses} \\
--host {build_host_arg(all_hosts, gpus_per_host)} \\
--allow-run-as-root \\
--display-map \\
--tag-output \\
-mca btl_tcp_if_include {if_name} \\
-mca oob_tcp_if_include {if_name} \\
-x NCCL_SOCKET_IFNAME={if_name} \\
--mca plm_rsh_no_tree_spawn 1 \\
-bind-to none -map-by slot \\
-mca pml ob1 -mca btl ^openib \\
-mca orte_abort_on_non_zero_status 1 \\
-x TENSORPACK_FP16=1 \\
-x NCCL_MIN_NRINGS={nccl_min_rings} -x NCCL_DEBUG=INFO \\
-x HOROVOD_CYCLE_TIME -x HOROVOD_FUSION_THRESHOLD \\
-x LD_LIBRARY_PATH -x PATH \\
--output-filename {model_dir}  \\
/usr/local/bin/python3.6 /mask-rcnn-tensorflow/MaskRCNN/train.py \
--logdir {log_dir} \
--fp16 \
--throughput_log_freq=2000 \
--images_per_epoch {images_per_epoch} \
--config \
MODE_FPN={mode_fpn} \
MODE_MASK={mode_mask} \
DATA.BASEDIR={train_data_dir} \
BACKBONE.RESNET_NUM_BLOCKS='{resnet_num_blocks}' \
BACKBONE.WEIGHTS={train_data_dir}/pretrained-models/{backbone_weights} \
BACKBONE.NORM={batch_norm} \
DATA.TRAIN='["{data_train}"]' \
DATA.VAL='("{data_val}",)' \
TRAIN.BATCH_SIZE_PER_GPU={batch_size_per_gpu} \
TRAIN.EVAL_PERIOD={eval_period} \
TRAIN.LR_EPOCH_SCHEDULE='{lr_epoch_schedule}' \
RPN.TOPK_PER_IMAGE=True \
PREPROC.PREDEFINED_PADDING=True \
TRAIN.GRADIENT_CLIP=0 \
TRAINER=horovod"""

    for key,item in hyperparamters.items():
        if key.startswith("config:"):
            hp=f" {key[7:]}={item}"
            mpirun_cmd+=hp
    
    if load_model:
        mpirun_cmd += f' --load {train_data_dir}/pretrained-models/{load_model}'
        
    print("--------Begin MPI Run Command----------")
    print(mpirun_cmd)
    print("--------End MPI Run Comamnd------------")
    exitcode = 0
    try:
        process = subprocess.Popen(mpirun_cmd, encoding='utf-8', cwd="/mask-rcnn-tensorflow",
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        while True:
            if process.poll() != None:
                break

            output = process.stdout.readline()
            if output:
                print(output.strip())

        exitcode = process.poll() 
        print(f"mpirun exit code:{exitcode}")
        exitcode = 0 
    except Exception as e:
        print("train exception occured", file=sys.stderr)
        exitcode = 1
        print(str(e), file=sys.stderr)
    finally:
        if copy_logs_to_model_dir:
            copy_files(log_dir, model_dir)

    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(exitcode)
    

if __name__ == "__main__":
    train()
