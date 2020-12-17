from smdebug.rules.rule import Rule
from smdebug import modes
import numpy as np

class check_accuracy(Rule):
    def __init__(self, base_trial, 
                 predictions='CrossEntropyLoss_0_input_0', 
                 labels='CrossEntropyLoss_0_input_1',
                 previous_accuracy=0.0,
                 threshold=0.05):
        super().__init__(base_trial)
        self.base_trial = base_trial
        self.labels = labels
        self.predictions = predictions 
        self.previous_accuracy = float(previous_accuracy)
        self.threshold = float(threshold)
        self.correct = 0
        self.samples = 0
        
    def invoke_at_step(self, step):
        
        # only get tensors from EVAL
        if self.base_trial._global_to_mode[step][0] == modes.EVAL:
            
            #convert global step into eval step
            step = self.base_trial._global_to_mode[step][1]

            #get tensors
            predictions = np.argmax(self.base_trial.tensor(self.predictions).value(step, mode=modes.EVAL), axis=1)
            labels = self.base_trial.tensor(self.labels).value(step, mode=modes.EVAL)
            
            #count correct predictions
            self.correct += np.sum(predictions == labels)
            
            #accuracy
            self.samples += predictions.shape[0]
            current_accuracy =  self.correct/self.samples
            
            if self.previous_accuracy - current_accuracy > self.threshold  : 
                self.logger.info(f"Step {step}: accuracy dropped by more than {self.threshold}. Current accuracy: {current_accuracy} Previous accuracy {self.previous_accuracy}")
                return True
            
            self.logger.info(f"Step {step}: current accuracy {current_accuracy}")

        return False
