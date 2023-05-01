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

class TestModel(ModelWrapper):
    def _initialize_model(self):
        f = nn.Sequential(
            nn.Linear(3,3),
            nn.ReLU(),
            nn.Linear(3,1),
        )
        f.load_state_dict(torch.load(self.model_path))
        return f


if __name__ == '__main__':
    f = TestModel()

    x = torch.Tensor([1,1,1])

    print(f(x).detach().numpy()[0])
    # torch.save(f.state_dict(), PATH)

