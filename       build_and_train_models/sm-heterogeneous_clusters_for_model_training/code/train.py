import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import logging
import sys
import os
import json

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
    
def main(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    train_kwargs = {'batch_size': args.batch_size,
                    'num_workers': args.num_data_workers,
                    'pin_memory': args.pin_memory
                   }
    logger.info ('Training job started...')
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.GaussianBlur(11)
        ])
    dataset = MyMNIST(batch_size=args.batch_size, iterations=args.iterations, root='./data', train=True,
                           transform=transform, download=True)
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
        if device=='cpu' or idx % 10 == 0:
            logger.info(
              f'{idx}: avg step time: {(time.perf_counter()-t)/idx}')
    logger.info('Training completed!')
    save_model(model, args.model_dir)

def save_model(model, model_dir):
    logger.info("Saving the model")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)
    return

def read_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch-size", type=int, default=4, 
                        help="Input batch size for training",)
    parser.add_argument("--iterations", type=int, default=10,
                        help="Based on no. of cpu per training instance",)
    parser.add_argument("--num-data-workers", type=int, default=1, metavar="N",
                        help="Based on no. of cpu per training instance type in data group",)
    parser.add_argument("--num-dnn-workers", type=int, default=1, metavar="N",
                        help="Based on no. of cpu per training instance type in dnn group, ideally should match to grpc-workers",)
    parser.add_argument("--grpc-workers", type=int, default=1, metavar="N",
                        help="No. of grpc server workers to start",)
    parser.add_argument("--pin-memory", type=bool, default=1, 
        help="pin to GPU memory (default: True)",)
    parser.add_argument("--seed",  type=int,  default=1,  
        help="random seed (default: 1)",)
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    #parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TESTING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--dispatcher_host", type=str)
    return parser.parse_args()

if __name__ == '__main__':
    main(read_args())
