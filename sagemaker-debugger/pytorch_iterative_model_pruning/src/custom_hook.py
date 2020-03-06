import smdebug.pytorch as smd

class CustomHook(smd.Hook):
             
    def forward_hook(self, module, inputs, outputs):
        module_name = self.module_maps[module]   
        self._write_inputs(module_name, inputs)
        
        #register outputs for backward pass. 
        outputs.register_hook(self.backward_hook(module_name + "_" + module._get_name() + "_output"))  
        self._write_outputs(module_name + "_" + module._get_name(), outputs)
        self.last_saved_step = self.step


