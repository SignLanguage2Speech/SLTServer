import torch
import torch.nn as nn

class ModelWrapper:
    def __init__(self, MODEL_PATH='model.pt'):
        self.model_path = MODEL_PATH
        self.model = self._initialize_model()
    def __call__(self, *input):
        return self.inference(*input)
    def _initialize_model(self):
        raise NotImplementedError("Implement in child classes")
    def inference(self, input):
        return self.model(input)