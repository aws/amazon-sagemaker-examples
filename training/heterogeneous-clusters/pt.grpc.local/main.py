import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import logging
import sys
import os

BATCH_SIZE = 8192 # Integer
ITERATIONS = 100 # No. of iterations in an epoc, must be multiple of 10s
DATALOADER_WORKERS = 4 # No. of workers, equals no. of CPUs of your local instance
MODEL_DIR = './'

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
    def __len__(self) -> int:
        import numpy as np
        size = BATCH_SIZE * ITERATIONS
        return size
    def __getitem__(self, index: int):
        return super(MyMNIST,self).__getitem__(index%len(self.data))
    
def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    train_kwargs = {'batch_size': BATCH_SIZE,
                    'num_workers': DATALOADER_WORKERS,
                    'pin_memory': True
                   }
    print ('Training job started...')
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.GaussianBlur(11)
        ])
    dataset = MyMNIST('./data', train=True, download=True,
                   transform=transform)
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
            print(
              f'{idx}: avg step time: {(time.perf_counter()-t)/idx}')
    print('Training completed!')
    save_model(model, MODEL_DIR)

def save_model(model, model_dir):
    logger.info("Saving the model")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)
    return
    
if __name__ == '__main__':
    main()
