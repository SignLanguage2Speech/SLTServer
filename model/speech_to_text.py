import whisper
import numpy as np
import ffmpeg
from model.model import ModelWrapper

class SpeechToText(ModelWrapper):
    def __init__(self):
        self.model = whisper.load_model("base")
        self.langauges = {
            "english": "en",
            "german": "de",
            "danish": "da"}
    def inference(self, waveform_input, language):
        return self.model.transcribe(
            waveform_input, 
            language=self.langauges[language], 
            patience=2, 
            beam_size=5)["text"]