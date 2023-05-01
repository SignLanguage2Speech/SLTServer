from model.utils import list2tensor, tensor2list, videobuffer2tensor
# from model.test import test_write_streamed_video, test_load_webm_from_bytes_into_tensor, test_load_webm_from_bytes

from mediaprocessing.video import write_video_tensor_to_mp4, webm_bytes_to_tensor
from mediaprocessing.audio import webm_to_waveform

from flask import Flask
from flask_sock import Sock
import pdb

class ModelServer:
    def __init__(self, slt_model, tts_model):
        self.app = Flask(__name__)
        self.sock = Sock(self.app)
        self.slt_model = slt_model
        self.tts_model = tts_model
        self.initialize_routes()
        self.initialize_sock()
        self.spoken_language = "german"
        self.signed_language_from = "english"
        self.signed_language_to = "english"
    
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
                video = webm_bytes_to_tensor(data)
                write_video_tensor_to_mp4(video) # ! FOR TESTING ONLY
                ws.send(data)
                del video
                del data
        @self.sock.route("/stt")
        def stt(ws): # Speech To Text. Receive .webm bytes (video) -> Send text
            while True:
                data = ws.receive()
                waveform = webm_to_waveform(data)
                result = self.tts_model(waveform, self.spoken_language)
                ws.send(result)
                del waveform
                del data

    def run(self):
        self.app.run(
            host="localhost",
            port='6969'
        )
    