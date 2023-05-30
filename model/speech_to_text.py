from whisper import load_model
from model.model import ModelWrapper

class SpeechToText(ModelWrapper):
    def __init__(self):
        self.model = load_model("base")
        self.langauges = {
            "US": "en",
            "UK": "en",
            "DE": "de",
            "DK": "da"}
    def inference(self, waveform_input, language_from):
        transcription = self.model.transcribe(
            waveform_input, 
            language=self.langauges[language_from], 
            patience=2, 
            beam_size=5)["text"]
        return transcription