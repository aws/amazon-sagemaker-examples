import torch
from torchvision import models
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import custom_hook
import smdebug.pytorch as smd
import numpy as np
import argparse
import logging
import sys
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

    
def loader(batch_size=128):

    # preprocessing for training data
    train_transforms = transforms.Compose([
                               transforms.RandomResizedCrop(224),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                           ])

    #preprocessing for validation data
    val_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ])
  
    # training data
    train_data = datasets.ImageFolder(os.environ['SM_CHANNEL_TRAIN'],
                              transform = train_transforms)

    # validation data
    val_data = datasets.ImageFolder(os.environ['SM_CHANNEL_TEST'],
                              transform = val_transforms)

    # train dataloader
    train_iterator = torch.utils.data.DataLoader(train_data, 
                                                 shuffle = True, 
                                                 num_workers=4,
                                                 batch_size = batch_size)
    # validation dataloader
    valid_iterator = torch.utils.data.DataLoader(val_data,
                                                 num_workers=4,
                                                 batch_size = batch_size)

    
    return train_iterator, valid_iterator

def train(epochs, batch_size, learning_rate):
    
    #load pruned model definition and weights
    checkpoint = torch.load("model_checkpoint")
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    
    #optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)    
    
    #count number of parameters 
    params = 0
    for parameter in model.parameters():
        parameter.requires_grad = True
        params += np.prod(parameter.size()) 
    
    #dataloader
    train_data_loader, val_data_loader = loader(batch_size)
 
    #loss
    criterion = torch.nn.CrossEntropyLoss()
    
    #define the custom hook
    hook = custom_hook.CustomHook.create_from_json_file()
    hook.register_module(model)
    hook.register_loss(criterion)
    
    #save number of parameters
    hook.save_scalar("parameters", params, sm_metric=True)
    
    #load data on the right context
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    #training loop
    for epoch in range(epochs):
        model.train()
        hook.set_mode(smd.modes.TRAIN)
    
        train_loss = 0
        for i, (batch, label) in enumerate(train_data_loader):
            batch = batch.to(device)
            label = label.to(device)
            model.zero_grad()
            output = model(batch)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
             
            train_loss += loss.item()
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, train_loss/(i+1)))
        hook.save_scalar("loss", train_loss/(i+1), sm_metric=True)
    
        #validation loop
        model.eval()
        hook.set_mode(smd.modes.EVAL)
        correct = 0
        total = 0
        for i, (batch, label) in enumerate(val_data_loader):
            batch = batch.to(device)
            label = label.to(device)
            
            output = model(batch)
            loss = criterion(output, label)
            _, predictions = output.max(1)
            correct += predictions.eq(label).sum().item()  
            total +=  predictions.size(0)
            
        print('acc:{:.4f}'.format(correct/total))
        hook.save_scalar("accuracy", correct/total, sm_metric=True)
        
if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)  
    parser.add_argument('--learning_rate', type=float, default=0.001)

    #parse arguments
    args, _ = parser.parse_known_args()
 
    #train model
    model = train(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate)
