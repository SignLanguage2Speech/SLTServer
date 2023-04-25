from model.model import TestModel
from server.server import ModelServer

if __name__ == '__main__':
    model = TestModel()
    model_server = ModelServer(model)
    model_server.run()