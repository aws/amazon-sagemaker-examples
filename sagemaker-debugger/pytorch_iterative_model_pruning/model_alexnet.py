import torch
from torchvision import models
import torchvision
import torch.nn as nn
import numpy as np
import smdebug
from smdebug import modes

#list of ordered tensor names 
activation_outputs = [
  'features.1_0_output_0',
  'features.4_0_output_0',
  'features.7_0_output_0',
  'features.9_0_output_0',
  'features.11_0_output_0']

weights =  [
  'AlexNet_features.0.weight',          
  'AlexNet_features.3.weight',
  'AlexNet_features.6.weight',
  'AlexNet_features.8.weight',
  'AlexNet_features.10.weight']

gradients = [
  'gradient/features.1_ReLU_output',
  'gradient/features.4_ReLU_output',
  'gradient/features.7_ReLU_output',
  'gradient/features.9_ReLU_output',
  'gradient/features.11_ReLU_output']

biases = [
 'AlexNet_features.0.bias',
 'AlexNet_features.3.bias',
 'AlexNet_features.6.bias',
 'AlexNet_features.8.bias',
 'AlexNet_features.10.bias']

classifier_weights = [
 'AlexNet_classifier.1.weight',
 'AlexNet_classifier.4.weight',
 'AlexNet_classifier.6.weight']

classifier_biases = [
 'AlexNet_classifier.1.bias',
 'AlexNet_classifier.4.bias',
 'AlexNet_classifier.6.bias']

#function to prune layers
def prune(model, activation_outputs, weights, biases, classifier_weights, classifier_biases, filters_list, trial, step):
    
    # dict that has a list of filters to be pruned per layer
    filters_dict = {}
    for layer_name, channel,_  in filters_list:
        if layer_name not in filters_dict:
            filters_dict[layer_name] = []
        filters_dict[layer_name].append(channel)
        
    counter = 0
    in_channels_dense = 0
    exclude_filters = None
    in_channels = 3
    
    #iterate over layers in the AlexNet model
    for layer1 in model.features.modules():  
        for layer2 in layer1.children():

            counter = counter + 1
            
            #check if current layer is a convolutional layer
            if isinstance(layer2, torch.nn.modules.conv.Conv2d):
                
                #remember the output channels of non-pruned convolution (needed for pruning first fc layer)
                in_channels_dense = layer2.out_channels

                #create key to find right weights/bias/filters for the corresponding layer
                key = "features." + str(counter) + "_0_output_0"
                index = activation_outputs.index(key)
                
                #get name of weight tensor
                weight_name = weights[index]
                
                #get weight values from last available training step
                weight = trial.tensor(weight_name).value(step, mode=modes.TRAIN)
                
                #get name of bias tensor
                bias_name = biases[index]
                
                #get vias values from last available training step
                bias = trial.tensor(bias_name).value(step, mode=modes.TRAIN)

                #we need to adjust the number of input channels,
                #if previous covolution has been pruned 
                if layer2.in_channels != in_channels:
                    layer2.in_channels = in_channels
                    weight  = np.delete(weight, exclude_filters, axis=1)
                    exclude_filters = None

                #if current layer is in the list of filters to be pruned
                if key in filters_dict:

                    print("Reduce output channels for layer",  counter, "from",  layer2.out_channels, "to", layer2.out_channels - len(filters_dict[key]))
                    
                    #set new output channels
                    layer2.out_channels = layer2.out_channels - len(filters_dict[key]) 
                    
                    #remove corresponding filters from weights and bias
                    #convolution weights have dimension: filter x channel x kernel x kernel
                    exclude_filters = filters_dict[key]
                    weight  = np.delete(weight, exclude_filters, axis=0)
                    bias =  np.delete(bias, exclude_filters, axis=0)
              
                #remember new size of output channels, because we need to prune subsequent convolution
                in_channels = layer2.out_channels  
                
                #set pruned weight and bias
                layer2.weight.data = torch.from_numpy(weight)
                layer2.bias.data = torch.from_numpy(bias)

              
    counter = 0
    
    #iterate over layers in the AlexNet model
    for layer1 in model.classifier.modules(): 
        for layer2 in layer1.children():
            
            #check if current layer is a fully connected layer
            if isinstance(layer2, torch.nn.Linear): 
                
                #get name of weight tensor 
                weight_name = classifier_weights[counter]
                
                #get weights from last training step
                weight = trial.tensor(weight_name).value(step, mode=modes.TRAIN)
                
                #get name of bias tensor
                bias_name = classifier_biases[counter]
                
                #get bias from last training step
                bias = trial.tensor(bias_name).value(step, mode=modes.TRAIN)
                
                #prune first fc layer
                if exclude_filters is not None:
                    
                    #in_channels_dense is the number of output channels of last non-pruned convolution layer
                    params = int(layer2.in_features/in_channels_dense)
          
                    #prune weights of first fc layer
                    indexes = []
                    for i in exclude_filters: 
                        indexes.extend(np.arange(i * params, (i+1)*params))
                        if indexes[-1] > weight.shape[1]:
                            indexes.extend(np.arange(weight.shape[1] - params , weight.shape[1]))   
                    weight  = np.delete(weight, indexes, axis=1)         
                    
                    print("Reduce weights for first linear layer from", layer2.in_features, "to", weight.shape[1])
                    
                    #set new in_features
                    layer2.in_features = weight.shape[1]
                    exclude_filters = None
               
                #set weights
                layer2.weight.data = torch.from_numpy(weight)
                
                #set bias
                layer2.bias.data = torch.from_numpy(bias)
                
                counter = counter + 1
                
    return model

