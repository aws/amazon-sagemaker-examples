import smdebug.pytorch as smd
from smdebug.core.modes import ModeKeys
import torch

class CustomHook(smd.Hook):
    
    #register input image for backward pass, to get image gradients
    def image_gradients(self, image):
        image.register_hook(self.backward_hook("image"))
        
    def forward_hook(self, module, inputs, outputs):
        module_name = self.module_maps[module]   
        self._write_inputs(module_name, inputs)
        
        #register outputs for backward pass. this is expensive, so we will only do it during EVAL mode
        if self.mode == ModeKeys.EVAL:
            outputs.register_hook(self.backward_hook(module_name + "_output"))
            
            #record running mean and var of BatchNorm layers
            if isinstance(module, torch.nn.BatchNorm2d):
                self._write_outputs(module_name + ".running_mean", module.running_mean)
                self._write_outputs(module_name + ".running_var", module.running_var)
            
        self._write_outputs(module_name, outputs)
        self.last_saved_step = self.step
