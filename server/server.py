from media_processing.video import write_video_tensor_to_mp4, write_single_frame_to_png, webm_bytes_to_tensor, VideoPipeline
from media_processing.audio import webm_to_waveform
from deep_translator import GoogleTranslator

from flask import Flask
from flask_sock import Sock
import json
import pdb

class ModelServer:
    def __init__(self, slt_model, stt_model):
        # k_t=0.333 # ! Temporal downsampling factor
        k_t=0.5 # ! Temporal downsampling factor
        # k_t=1. # ! Temporal downsampling factor
        self.app = Flask(__name__)
        self.sock = Sock(self.app)
        self.pipeline = VideoPipeline(k_t=k_t)
        self.slt_model = slt_model # Sign Language Translation
        self.stt_model = stt_model # Speech To Text
        self.initialize_routes()
        self.initialize_sock()
        self.spoken_language = "US"
        self.signed_language_from = "US"
        self.signed_language_to = "US"
        self.performance = 3
        self.google_languages = {
            "US": "en",
            "UK": "en",
            "DE": "de",
            "DK": "da"}
    
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
                # write_video_tensor_to_mp4(video)
                video, num_frames = self.pipeline(video, to_file=False, output_length=True)
                # pdb.set_trace()
                # write_video_tensor_to_mp4(processed_video, w=224, h=224, fps=30, OUT_FILE_PATH='processed_output.mp4') # ! FOR TESTING ONLY => write processed video
                y = self.slt_model(video, num_frames)[0]
                if self.spoken_language != self.signed_language_from:
                    y = GoogleTranslator(source=self.google_languages[self.signed_language_from], target=self.google_languages[self.spoken_language]).translate(y)
                del video
                print("prediction", y)
                ws.send(y)
        @self.sock.route("/stt")
        def stt(ws): # Speech To Text. Receive .webm bytes (video) -> Send text
            while True:
                data = ws.receive()
                waveform = webm_to_waveform(data)
                y = self.stt_model(waveform, self.spoken_language)
                if self.spoken_language != self.signed_language_to:
                    y = GoogleTranslator(source=self.google_languages[self.spoken_language], target=self.google_languages[self.signed_language_to]).translate(y)
                print(self.spoken_language, self.signed_language_to)
                ws.send(y)
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

        @self.sock.route("/set_performance")
        def set_performance(ws): # Set Performance
            while True:
                data = ws.receive()
                performance: dict = json.loads(data)
                self.performance = performance["performance"]
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
            port='8080'
        )
