from model.utils import list2tensor, tensor2list, videobuffer2tensor
from model.test import test_write_streamed_video, test_load_webm_from_bytes

from flask import Flask
from flask_sock import Sock

class ModelServer:
    def __init__(self, model):
        self.app = Flask(__name__)
        self.sock = Sock(self.app)
        self.model = model
        self.initialize_routes()
        self.initialize_sock()
    
    def initialize_routes(self):
        @self.app.route('/1')
        def inf1():
            x = list2tensor([1,0,0])
            return tensor2list(self.model(x))
    
    def initialize_sock(self):
        @self.sock.route("/echo")
        def echo(ws):
            while True:
                data = ws.receive()
                print(data)
                ws.send(data)
        @self.sock.route("/inference")
        def inference(ws):
            while True:
                data = ws.receive()
                # print(data)
                # tns_data = videobuffer2tensor(data)
                # test_write_streamed_video(tns_data)
                test_load_webm_from_bytes(data)
                ws.send(data)

    def run(self):
        self.app.run(
            host="localhost",
            port='6969'
        )
    