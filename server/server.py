from media_processing.video import write_video_tensor_to_mp4, write_webm_bytes_to_file, webm_bytes_to_tensor, get_video_dims_from_webm_bytes, VideoPipeline
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
            # i = 16
            while True:
                data = ws.receive()
                # write_webm_bytes_to_file(data, OUT_FILE_PATH=f"./experiment-data/experiment_speech_vids/speech{i}.webm")
                # i+=1
                width, height = get_video_dims_from_webm_bytes(data)
                self.pipeline.W_in, self.pipeline.H_in = width, height
                video = webm_bytes_to_tensor(data, device='cpu', width=width, height=height)
                del data
                
                video, num_frames = self.pipeline(video, to_file=False, output_length=True)
                y = self.slt_model(video, num_frames)[0]
                if self.spoken_language != self.signed_language_from:
                    y = GoogleTranslator(source=self.google_languages[self.signed_language_from], target=self.google_languages[self.spoken_language]).translate(y)
                print("prediction", y)
                del video
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
