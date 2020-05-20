from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os

import smdebug.pytorch as smd
from smdebug import modes
from smdebug.core.modes import ModeKeys
from custom_hook import CustomHook

def get_dataloaders(batch_size_train, batch_size_val):
    train_transform =  transforms.Compose([
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
    
    val_transform = transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ])

    dataset = datasets.ImageFolder(os.environ['SM_CHANNEL_TRAIN'], train_transform)

    train_dataloader = torch.utils.data.DataLoader(dataset, 
                                                  batch_size=batch_size_train,
                                                  shuffle=True)

    dataset = datasets.ImageFolder(os.environ['SM_CHANNEL_TEST'], val_transform)

    val_dataloader = torch.utils.data.DataLoader( dataset, 
                                                  batch_size=batch_size_val,
                                                  shuffle=False)

    return train_dataloader, val_dataloader

def relu_inplace(model):
    for child_name, child in model.named_children():
        if isinstance(child, torch.nn.ReLU):
            setattr(model, child_name, torch.nn.ReLU(inplace=False))
        else:
            relu_inplace(child)
            
def train_model(epochs, batch_size_train, batch_size_val):
    
    #check if GPU is available and set context
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #get pretrained ResNet model
    model = models.resnet18(pretrained=True)
    
    #replace inplace operators
    relu_inplace(model)

    nfeatures = model.fc.in_features

    #traffic sign dataset has 43 classes
    model.fc = nn.Linear(nfeatures, 43)

    #copy model to GPU or CPU
    model = model.to(device)

    # loss for multi label classification
    loss_function = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

    #configure smdebug hook: 
    #save all iterations from validation phase
    #save only first iteration from training phase
    save_config = smd.SaveConfig(mode_save_configs={
        smd.modes.TRAIN: smd.SaveConfigMode(save_steps=[0]),
        smd.modes.EVAL: smd.SaveConfigMode(save_interval=1)
    })
    
    #create custom hook that has a customized forward function, so that we can get gradients of outputs   
    hook = CustomHook(args.smdebug_dir, save_config=save_config, include_regex='.*bn|.*bias|.*downsample|.*ResNet_input|.*image|.*fc_output' )
    
    #register hook
    hook.register_module(model)  
    
    #get the dataloaders for train and test data
    train_loader, val_loader = get_dataloaders(batch_size_train, batch_size_val)
    
    #training loop
    for epoch in range(epochs):
        
        epoch_loss = 0
        epoch_acc = 0
        
        #set hook training phase
        hook.set_mode(modes.TRAIN)
        model.train()
        
        for inputs, labels in train_loader: 
            inputs = inputs.to(device).requires_grad_()
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(inputs)
            
            #get predictions
            _, preds = torch.max(outputs, 1)
            
            #compute loss
            loss = loss_function(outputs, labels)
            
            #backward pass
            loss.backward()
            
            #optimize parameters
            optimizer.step()
            
            # statistics
            epoch_loss += loss.item() 
            epoch_acc += torch.sum(preds == labels.data)

        #set hook validation phase
        hook.set_mode(modes.EVAL)
        model.eval()
        
        for inputs, labels in val_loader: 
         
            inputs = inputs.to(device).requires_grad_()
            hook.image_gradients(inputs)
            
            model.eval()
            
            #forward pass
            outputs = model(inputs)
            
            #get prediction
            predicted_class = outputs.data.max(1, keepdim=True)[1]
            agg = 0
            for i in range(outputs.shape[0]):
                agg += outputs[i,predicted_class[i]]
            model.zero_grad()
            
            #compute gradients with respect to outputs 
            agg.backward()

        print('Epoch {}/{} Loss: {:.4f} Acc: {:.4f}'.format(
            epoch, epochs - 1, epoch_loss, epoch_acc))

    return model

if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size_train', type=int, default=64)  
    parser.add_argument('--batch_size_val', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    
    # Data, model, and output directories
    parser.add_argument('--smdebug_dir', type=str, default=None)

    #parse arguments
    args, _ = parser.parse_known_args()
    

    #train model
    model = train_model(epochs=args.epochs, batch_size_train=args.batch_size_train, batch_size_val=args.batch_size_val)
