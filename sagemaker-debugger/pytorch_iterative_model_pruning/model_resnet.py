import torch
from torchvision import models
import torchvision
import torch.nn as nn
import numpy as np
import smdebug
from smdebug import modes

#list of ordered tensor names 
activation_outputs = [
#'relu_ReLU_output_0',
'layer1.0.relu_0_output_0',
'layer1.1.relu_0_output_0',
'layer2.0.relu_0_output_0',
'layer2.1.relu_0_output_0',
'layer3.0.relu_0_output_0',
'layer3.1.relu_0_output_0',
'layer4.0.relu_0_output_0',
'layer4.1.relu_0_output_0'
]

gradients = [
#'gradient/relu_ReLU_output',
'gradient/layer1.0.relu_ReLU_output',
'gradient/layer1.1.relu_ReLU_output',
'gradient/layer2.0.relu_ReLU_output',
'gradient/layer2.1.relu_ReLU_output',
'gradient/layer3.0.relu_ReLU_output',
'gradient/layer3.1.relu_ReLU_output',
'gradient/layer4.0.relu_ReLU_output',
'gradient/layer4.1.relu_ReLU_output'
]


#function to prune layers
def prune(model, filters_list, trial, step):
    
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
    exclude = False
      #iterate over layers in the ResNet model
    for named_module in model.named_modules():
    
        layer_name = named_module[0]
        layer = named_module[1]
        
        #check if current layer is a convolutional layer
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
     
            #remember the output channels of non-pruned convolution (needed for pruning first fc layer)
            in_channels_dense = layer.out_channels

            #create key to find right weights/bias/filters for the corresponding layer
            weight_name = "ResNet_" + layer_name + ".weight"
        
            #get weight values from last available training step
            weight = trial.tensor(weight_name).value(step, mode=modes.TRAIN)
            
            #we need to adjust the number of input channels,
            #if previous covolution has been pruned 
            #print( "current:", layer.in_channels, "previous", in_channels, layer_name, weight_name)
            if 'conv1' in layer_name or 'conv2' in layer_name:
                if layer.in_channels != in_channels:
                    layer.in_channels = in_channels
                    weight  = np.delete(weight, exclude_filters, axis=1)
                    exclude_filters = None
                    
            #if current layer is in the list of filters to be pruned
            if "conv1" in layer_name: 
                layer_id = layer_name.strip("conv1")
                for key in filters_dict:  

                    if len(layer_id) > 0 and layer_id in key:
   
                        print("Reduce output channels for conv layer",  layer_id, "from",  layer.out_channels, "to", layer.out_channels - len(filters_dict[key]))

                        #set new output channels
                        layer.out_channels = layer.out_channels - len(filters_dict[key]) 

                        #remove corresponding filters from weights and bias
                        #convolution weights have dimension: filter x channel x kernel x kernel
                        exclude_filters = filters_dict[key]
                        weight  = np.delete(weight, exclude_filters, axis=0)
                        break
                              
            #remember new size of output channels, because we need to prune subsequent convolution
            in_channels = layer.out_channels  

            #set pruned weight and bias
            layer.weight.data = torch.from_numpy(weight)
            
        if isinstance(layer,  torch.nn.modules.batchnorm.BatchNorm2d):   
          
            #get weight values from last available training step
            weight_name = "ResNet_" + layer_name + ".weight"
            weight = trial.tensor(weight_name).value(step, mode=modes.TRAIN)
            
            #get bias values from last available training step
            bias_name = "ResNet_" + layer_name + ".bias"
            bias = trial.tensor(bias_name).value(step, mode=modes.TRAIN)
            
            #get running_mean values from last available training step
            mean_name = layer_name + ".running_mean_output_0"
            mean = trial.tensor(mean_name).value(step, mode=modes.TRAIN)
            
            #get running_var values from last available training step
            var_name = layer_name + ".running_var_output_0"
            var = trial.tensor(var_name).value(step, mode=modes.TRAIN)
            
            #if current layer is in the list of filters to be pruned
            if "bn1" in layer_name: 
                layer_id = layer_name.strip("bn1")
                for key in filters_dict:  
                    if len(layer_id) > 0 and layer_id in key:

                        print("Reduce bn layer",  layer_id, "from",  weight.shape[0], "to", weight.shape[0] - len(filters_dict[key]))

                        #remove corresponding filters from weights and bias
                        #convolution weights have dimension: filter x channel x kernel x kernel
                        exclude_filters = filters_dict[key]
                        weight  = np.delete(weight, exclude_filters, axis=0)
                        bias =  np.delete(bias, exclude_filters, axis=0)
                        mean =  np.delete(mean, exclude_filters, axis=0)
                        var  =  np.delete(var, exclude_filters, axis=0)
                        break

            #set pruned weight and bias
            layer.weight.data = torch.from_numpy(weight)
            layer.bias.data = torch.from_numpy(bias)
            layer.running_mean.data = torch.from_numpy(mean)
            layer.running_var.data = torch.from_numpy(var)
            layer.num_features = weight.shape[0]
            in_channels = weight.shape[0]
            
        if isinstance(layer, torch.nn.modules.linear.Linear):

            #get weight values from last available training step
            weight_name = "ResNet_" + layer_name + ".weight"
            weight = trial.tensor(weight_name).value(step, mode=modes.TRAIN)
            
            #get bias values from last available training step
            bias_name = "ResNet_" + layer_name + ".bias"
            bias = trial.tensor(bias_name).value(step, mode=modes.TRAIN)
            
            #prune first fc layer
            if exclude_filters is not None:
                 #in_channels_dense is the number of output channels of last non-pruned convolution layer
                params = int(layer.in_features/in_channels_dense)

                #prune weights of first fc layer
                indexes = []
                for i in exclude_filters: 
                    indexes.extend(np.arange(i * params, (i+1)*params))
                    if indexes[-1] > weight.shape[1]:
                        indexes.extend(np.arange(weight.shape[1] - params , weight.shape[1]))   
                weight  = np.delete(weight, indexes, axis=1)         

                print("Reduce weights for first linear layer from", layer.in_features, "to", weight.shape[1])
                 #set new in_features
                layer.in_features = weight.shape[1]
                exclude_filters = None

            #set weights
            layer.weight.data = torch.from_numpy(weight)

            #set bias
            layer.bias.data = torch.from_numpy(bias)
                
    return model

