import whisper
from deep_translator import GoogleTranslator
import numpy as np
import ffmpeg
from model.model import ModelWrapper

class SpeechToText(ModelWrapper):
    def __init__(self):
        self.model = whisper.load_model("base")
        self.langauges = {
            "US": "en",
            "UK": "en",
            "DE": "de",
            "DK": "da"}
    def inference(self, waveform_input, language_from, language_to):
        transcription = self.model.transcribe(
            waveform_input, 
            language=self.langauges[language_from], 
            patience=2, 
            beam_size=5)["text"]
        if language_from == language_to:
            return transcription
        return GoogleTranslator(source=self.langauges[language_from], target=self.langauges[language_to]).translate(transcription)