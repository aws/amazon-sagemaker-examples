from tensorflow.data.experimental.service import DispatchServer, WorkerServer, DispatcherConfig, WorkerConfig

def wait_for_shutdown_signal(dispatcher, workers):
    SHUTDOWN_PORT = 16000
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', SHUTDOWN_PORT))
    s.listen(1)
    print('Awaiting shutdown signal on port {}'.format(SHUTDOWN_PORT))
    conn, addr = s.accept()
    print('Received shutdown signal from: ', addr)
    try:
        conn.close()
        s.close()
    except Exception as e:
        print(e)
    
    if dispatcher is not None: # dispatcher runs only on the 1st data instance
        print('Stopping dispatcher.')
        dispatcher._stop()
        print('Joining dispatcher')
        dispatcher.join()
    
    for i,worker in enumerate(workers, start=0):
        print(f'Stopping worker {i}')
        worker._stop()
        print(f'Joining worker {i}')
        worker.join()

def create_worker(workerIndex : int, dispatcher_host : str, current_host : str) -> WorkerServer:
    port = 6001 + workerIndex
    w_config = WorkerConfig(port=port,
        dispatcher_address=f'{dispatcher_host}:6000',
        worker_address=f'{current_host}:{port}')
    print(f'Starting tf.data.service WorkerServer {w_config}')
    worker = WorkerServer(w_config)
    return worker

def start_dispatcher_and_worker(dispatcher_host : str, current_host : str, num_of_data_workers : int):
    assert(dispatcher_host is not None)
    
    if current_host == dispatcher_host:
        print(f'starting Dispatcher (dispatcher_host={dispatcher_host})')
        d_config = DispatcherConfig(port=6000)
        dispatcher = DispatchServer(d_config)
    else:
        dispatcher = None

    workers = [ create_worker(i, dispatcher_host, current_host) for i in range(num_of_data_workers) ]
    print(f'Finished starting dispatcher and {num_of_data_workers} workers')
    
    wait_for_shutdown_signal(dispatcher, workers)


"This function read mode command line argument"
def read_args():
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--dispatcher_host", type=str)
    parser.add_argument("--current_host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--num_of_data_workers", type=int)
    args, unknown = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = read_args()
    start_dispatcher_and_worker(args.dispatcher_host, args.current_host, args.num_of_data_workers)