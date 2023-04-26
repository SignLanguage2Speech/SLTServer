from model.utils import list2tensor, tensor2list, videobuffer2tensor
# from model.test import test_write_streamed_video, test_load_webm_from_bytes_into_tensor, test_load_webm_from_bytes

from signals.video import write_video_tensor_to_mp4, convert_webm_bytes_to_tensor

from flask import Flask
from flask_sock import Sock
from flask import request
import pdb
import json

class ModelServer:
    def __init__(self, slt_model, stt_model):
        self.app = Flask(__name__)
        self.sock = Sock(self.app)
        self.slt_model = slt_model
        self.stt_model = stt_model
        self.initialize_routes()
        self.initialize_sock()
        self.spoken_language = "US"
        self.signed_language_from = "US"
        self.signed_language_to = "US"
    
    def initialize_routes(self):
        @self.app.route('/inference_test')
        def inf1():
            x = list2tensor([1,0,0])
            return tensor2list(self.slt_model(x))
    
    def initialize_sock(self):
        @self.sock.route("/slt")
        def slt(ws): # Sign Language Translation. Receive .webm bytes (video) -> Send text
            while True:
                data = ws.receive()
                video = convert_webm_bytes_to_tensor(data)
                write_video_tensor_to_mp4(video) # ! FOR TESTING ONLY
                ws.send(data)
                del data
        @self.sock.route("/stt")
        def stt(ws): # Speech To Text. Receive .webm bytes (video) -> Send text
            while True:
                data = ws.receive()
                result = self.stt_model.inference(data,self.spoken_language, self.signed_language_to)
                print(self.spoken_language, self.signed_language_to)
                # if "Copyright WDR 2020" in result:
                #     result = ""
                ws.send(result)
                del data

        @self.sock.route("/change_language")
        def change_language(ws): # Change language
            while True:
                data = ws.receive()
                languages: dict = json.loads(data)
                self.spoken_language = languages.get("spl", "US")
                self.signed_language_from = languages.get("silf", "US")
                self.signed_language_to = languages.get("silt", "US")
                del data

    def run(self):
        self.app.run(
            host="localhost",
            port='6969'
        )
    