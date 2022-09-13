import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import grpc
import dataset_feed_pb2_grpc
import dataset_feed_pb2
import logging
import sys
import json
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# Decode binary data from SM_CHANNEL_TRAINING
# Decode and preprocess data
# Create map dataset
class RemoteDataset(torch.utils.data.IterableDataset):
    '''
    An iterable PyTorch dataset that opens a connection to the
    gRPC server and reads from a stream of data batches 
    '''

    def __init__(self, batch_size, iterations):
        self.batch_size = batch_size
        self.iterations = iterations

        
    def __len__(self) -> int:
        size = self.batch_size * self.iterations
        return size

    def get_stub(self):
        host = 'localhost'
        channel = grpc.insecure_channel(f'{host}:6000',
                    # overwrite the default max message length
                    options=[('grpc.max_receive_message_length',
                               200 * 1024 * 1024)])

        try:
        #    print('Waiting for gRPC data server to be ready...')
            grpc.channel_ready_future(channel).result(timeout=30)
        except grpc.FutureTimeoutError:
            print('ERROR: Timeout connecting to gRPC data server. Check that it is running.')
            raise
        #print('Connected to gRPC data server.')

        return dataset_feed_pb2_grpc.DatasetFeedStub(channel,)


    def __iter__(self):
        import numpy as np
        
        examples = self.get_stub().get_examples(dataset_feed_pb2.Dummy())
        for s in examples:
            image = torch.tensor(np.frombuffer(s.image, 
                              dtype=np.float32)).reshape(
                                       [self.batch_size, 1, 28, 28])
            label = torch.tensor(np.frombuffer(s.label, 
                              dtype=np.int8)).reshape(
                                       [self.batch_size]).type(torch.int64)
            yield image, label
    

    # def shutdown_remote(self):
    #     print('Calling remote server to shutdown')
    #     self.get_stub().shutdown(dataset_feed_pb2.Dummy())

def shutdown_data_service():
    SHUTDOWN_PORT = 16000
    print('Shutting down data service via port {}'.format(SHUTDOWN_PORT))
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('localhost', SHUTDOWN_PORT))
    s.close()
    
def main(args):
    print ('Training job started...')
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    train_kwargs = {'batch_size': None, #the data is already batched
                    'num_workers': args.num_dnn_workers, #no. of cpus in dnn instance type
                    'pin_memory': args.pin_memory,
                   }
    dataset = RemoteDataset(args.batch_size, args.iterations)
    train_loader = torch.utils.data.DataLoader(dataset,
                                               **train_kwargs)
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters())
    model.train()
    t = time.perf_counter()
    for idx, (data, target) in enumerate(train_loader, start=1):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if device.type == 'cpu'or idx % 10 == 0:
            logger.info(
              f'{idx}: avg step time: {(time.perf_counter()-t)/idx}')

        # TODO: exit the loop through the iterator stopping by itself
        if idx*args.batch_size==(dataset.__len__()):
            break
    save_model(model, args.model_dir)
    print ('Training job completed!')
    shutdown_data_service()

def save_model(model, model_dir):
    logger.info("Saving the model")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)
    return

"This function read mode command line argument"
def read_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4, metavar="N",
                        help="Input batch size for training",)
    parser.add_argument("--num-dnn-workers", type=int, default=1, metavar="N",
                        help="Based on no. of cpu per training instance",)
    parser.add_argument("--pin-memory", type=bool, default=1, metavar="N", 
        help="Pin to GPU memory (default: True)",)
    parser.add_argument("--seed",  type=int,  default=1,  metavar="S",  
        help="random seed (default: 1)",)
    parser.add_argument("--model-dir", type=str)
    parser.add_argument("--iterations", type=int, default=10, metavar="N",
                        help="The number of iterations per epoch (multiples of 10)",)
    parser.add_argument("--first_data_host", type=str)
    args, unknown = parser.parse_known_args()
    return args

if __name__ == '__main__':
    main(read_args())
