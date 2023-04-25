from model.utils import list2tensor, tensor2list, videobuffer2tensor
from model.test import test_write_streamed_video, test_load_webm_from_bytes

from flask import Flask
from flask_sock import Sock
import pdb

class ModelServer:
    def __init__(self, model, tts_model):
        self.app = Flask(__name__)
        self.sock = Sock(self.app)
        self.model = model
        self.tts_model = tts_model
        self.initialize_routes()
        self.initialize_sock()
        self.spoken_language = "english"
        self.signed_language_from = "english"
        self.signed_language_to = "english"
    
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
        @self.sock.route("/stt")
        def stt(ws):
            while True:
                data = ws.receive()
                out = self.tts_model.inference(data,self.spoken_language)
                ws.send(out)

    def run(self):
        self.app.run(
            host="localhost",
            port='6969'
        )
    