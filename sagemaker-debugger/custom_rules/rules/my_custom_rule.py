# First Party
from smdebug.rules.rule import Rule


class CustomGradientRule(Rule):
    def __init__(self, base_trial, threshold=10.0):
        super().__init__(base_trial)
        self.threshold = float(threshold)

    def set_required_tensors(self, step):
        for tname in self.base_trial.tensor_names(collection="gradients"):
            self.req_tensors.add(tname, steps=[step])

    def invoke_at_step(self, step):
        for t in self.req_tensors.get():
            abs_mean = t.reduction_value(step, "mean", abs=True)
            if abs_mean > self.threshold:
                return True
        return False
    