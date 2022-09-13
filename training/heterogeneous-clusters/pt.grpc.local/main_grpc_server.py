import multiprocessing as mp
from concurrent import futures

import grpc
import torch
from torchvision import datasets, transforms

import dataset_feed_pb2
import dataset_feed_pb2_grpc
import logging
import sys

# Logging initialization
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# The following class implements the data feeding service
class DatasetFeedService(dataset_feed_pb2_grpc.DatasetFeedServicer):
    def __init__(self, q, kill_event):
        '''
        param q: A shared queue containing data batches
        param kill: Kill event for graceful shutdown
        '''
        self.q = q
        self.kill_event = kill_event


    def get_examples(self, request, context):
        while True:
            #print('DEBUG: get_examples')
            example = self.q.get()
            yield dataset_feed_pb2.Example(image=example[0], 
                                       label=example[1])


    def shutdown(self, request, context):
        print("Received shutdown request - Not implemented")
        # from main_grpc_client import shutdown_data_service
        # shutdown_data_service()
        context.set_code(grpc.StatusCode.OK)
        context.set_details('Shutting down')
        return dataset_feed_pb2.Dummy()


# The data loading and preprocessing logic.
# We chose to keep the existing logic unchanged, just instead
# of feeding the model, the dataloader feeds a shared queue
class MyMNIST(datasets.MNIST):
    '''
    A personalized extension of the MNIST class in which we
    modify the __len__ operation to return the maximum value
    of int32 so that we do not run out of data. 
    '''

    def __init__(self, batch_size : int, iterations : int, **kwargs):
        
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.iterations = iterations

    def __len__(self) -> int:
        size = self.batch_size * self.iterations
        return size

    def __getitem__(self, index: int):
        return super(MyMNIST, self).__getitem__(index % len(self.data))


def fill_queue(q,kill, args):
    train_kwargs = {'batch_size': args.batch_size,
                    'num_workers': args.num_data_workers}
    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.GaussianBlur(11)
            ])
    dataset = MyMNIST(batch_size=args.batch_size, iterations=args.iterations, root='./data', train=True,
                           transform=transform, download=True)
    loader = torch.utils.data.DataLoader(dataset, **train_kwargs)
    for batch_idx, (data, target) in enumerate(loader):
        if kill.is_set():
            print('kill signal received, exiting fill_queue')
            break
        added = False
        while not added and not kill.is_set():
            try:
                # convert the data to bytestrings and add to queue               
                q.put((data.numpy().tobytes(),
                       target.type(torch.int8).numpy().tobytes()),
                       timeout=1)
                #print(f'DEBUG: Added example to queue')
                added = True
            except:
                continue
    print('Finished filling queue with dataset.')


def start(kill_event, args):
    q = mp.Queue(maxsize=32)
    queuing_process = mp.Process(target=fill_queue, args=(q, kill_event, args))
    queuing_process.start()
    print('Started queuing process.')

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.grpc_workers))
    dataset_feed_pb2_grpc.add_DatasetFeedServicer_to_server(
        DatasetFeedService(q, kill_event), server)
    server.add_insecure_port('[::]:6000')
    server.start()
    print('gRPC Data Server started at port 6000.')
    return queuing_process,server


def shutdown(queuing_process, grpc_server):
    print('Shutting down...')
    print('Stopping gRPC server...')
    grpc_server.stop(2).wait()
    print('Stopping queuing process...')
    queuing_process.join(1)
    queuing_process.terminate()
    print('Shutdown done.')
    import os, time
    os.system('kill %d' % os.getpid())
    time.sleep(2)
    os.system('kill -9 %d' % os.getpid())


def wait_for_shutdown_signal():
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


def serve(args):
    kill_event = mp.Event() # an mp.Event for graceful shutdown
    queue_data_loader_process, grpc_server = start(kill_event, args)
    wait_for_shutdown_signal()
    kill_event.set()
    shutdown(queue_data_loader_process, grpc_server)


"This function read mode command line argument"
def read_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4, metavar="N",
                        help="Input batch size for training",)
    parser.add_argument("--num-data-workers", type=int, default=1, metavar="N",
                        help="Based on no. of cpu per training instance",)
    parser.add_argument("--iterations", type=int, default=10, metavar="N",
                        help="The number of iterations per epoch (multiples of 10)",)
    parser.add_argument("--grpc-workers", type=int, default=1, metavar="N",
                        help="No. of gRPC server workers",)
    parser.add_argument("--first_data_host", type=str)
    args, unknown = parser.parse_known_args()
    return args


if __name__ == "__main__":
    serve(read_args())
