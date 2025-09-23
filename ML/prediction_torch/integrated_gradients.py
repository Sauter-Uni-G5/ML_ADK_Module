from captum.attr import IntegratedGradients
import torch
import numpy as np

class IG_module():
    def __init__(self, model, baseline, target, steps=200):
        self.model = model.eval()
        self.steps = steps
        self.baseline = baseline
        self.target = torch.tensor(target, dtype=torch.long) if target is not None else None
        self.ig = IntegratedGradients(self.model)
    
    def attribute(self, input):
        input_tensor = torch.tensor(input, dtype=torch.float32)
        attributions, delta = self.ig.attribute(
            input_tensor,
            baselines=self.baseline,
            target=self.target,
            n_steps=self.steps,
            internal_batch_size=None,
            return_convergence_delta=True
        )
        return attributions.detach().numpy(), delta.detach().numpy()