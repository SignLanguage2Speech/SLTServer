from model.utils import list2tensor, tensor2list, videobuffer2tensor
# from model.test import test_write_streamed_video, test_load_webm_from_bytes_into_tensor, test_load_webm_from_bytes

from media_processing.video import write_video_tensor_to_mp4, webm_bytes_to_tensor, VideoPipeline
from media_processing.audio import webm_to_waveform

from flask import Flask
from flask_sock import Sock
import json

class ModelServer:
    def __init__(self, slt_model, stt_model):
        self.app = Flask(__name__)
        self.sock = Sock(self.app)
        self.pipe = VideoPipeline()
        self.slt_model = slt_model # Sign Language Translation
        self.stt_model = stt_model # Speech To Text
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
                video = webm_bytes_to_tensor(data)
                processed_video = self.pipe(video, to_file=True)
                write_video_tensor_to_mp4(video)                                                                       # ! FOR TESTING ONLY => write unaltered video
                write_video_tensor_to_mp4(processed_video, w=224, h=224, fps=30, OUT_FILE_PATH='processed_output.mp4') # ! FOR TESTING ONLY => write processed video
                ws.send(data)
                del video
                del data
        @self.sock.route("/stt")
        def stt(ws): # Speech To Text. Receive .webm bytes (video) -> Send text
            while True:
                data = ws.receive()
                waveform = webm_to_waveform(data)
                result = self.stt_model(waveform, self.spoken_language, self.signed_language_to)
                print(self.spoken_language, self.signed_language_to)
                ws.send(result)
                del waveform
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
    