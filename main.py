from model.model import TestModel
from server.server import ModelServer
from model.utils import Logger as LOG

if __name__ == '__main__':
    LOG.toggle_dbg(debug=True)
    # LOG.toggle_dbg(debug=False)
    model = TestModel()
    model_server = ModelServer(model)
    model_server.run()