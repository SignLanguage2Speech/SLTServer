from media_processing.video import write_video_tensor_to_mp4, webm_bytes_to_tensor, VideoPipeline
from media_processing.audio import webm_to_waveform

from flask import Flask
from flask_sock import Sock
import json

class ModelServer:
    def __init__(self, slt_model, stt_model):
        self.app = Flask(__name__)
        self.sock = Sock(self.app)
        self.pipeline = VideoPipeline()
        self.slt_model = slt_model # Sign Language Translation
        self.stt_model = stt_model # Speech To Text
        self.initialize_routes()
        self.initialize_sock()
        self.spoken_language = "US"
        self.signed_language_from = "US"
        self.signed_language_to = "US"
    
    def initialize_routes(self):
        @self.app.route('/hello')
        def inf1():
            return "Hallo!"
    
    def initialize_sock(self):
        @self.sock.route("/slt")
        def slt(ws): # Sign Language Translation. Receive .webm bytes (video) -> Send text
            while True:
                data = ws.receive()
                video = webm_bytes_to_tensor(data, device='cpu')
                del data
                # processed_video, num_frames = self.pipeline(video, to_file=True, output_length=True)
                video, num_frames = self.pipeline(video, to_file=False, output_length=True)
                # write_video_tensor_to_mp4(video)                                                                       # ! FOR TESTING ONLY => write unaltered video
                # write_video_tensor_to_mp4(processed_video, w=224, h=224, fps=30, OUT_FILE_PATH='processed_output.mp4') # ! FOR TESTING ONLY => write processed video
                y = self.slt_model(video, num_frames)
                del video
                print("prediction", y)
                ws.send(y[0])
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

        @self.sock.route("/check")
        def check(ws): # Check connection
            while True:
                data = ws.receive()
                ws.send(200)
                del data

    def run(self):
        self.app.run(
            host="localhost",
            port='6969'
        )
    