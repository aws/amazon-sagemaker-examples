import smdebug.pytorch as smd
import torch

class CustomHook(smd.Hook):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.written_tensors = {}
            
    def forward_hook(self, module, inputs, outputs):
        module_name = self.module_maps[module]   
       
        #register outputs for backward pass. 
        outputs.register_hook(self.backward_hook(module_name + "_" + module._get_name() + "_output"))  
        if isinstance(module, torch.nn.BatchNorm2d):
            self._write_outputs(module_name + ".running_mean", module.running_mean)
            self._write_outputs(module_name + ".running_var", module.running_var)
            
        if self.step not in self.written_tensors:
            self.written_tensors[self.step] = {}
        if module_name not in self.written_tensors[self.step]:
            self.written_tensors[self.step][module_name] = 0   
        else:
            self.written_tensors[self.step][module_name] += 1
            
        module_name = module_name + "_" + str(self.written_tensors[self.step][module_name])
        
        self._write_inputs(module_name, inputs)
        self._write_outputs(module_name, outputs)
        self.last_saved_step = self.step


