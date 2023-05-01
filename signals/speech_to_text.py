import whisper
from deep_translator import GoogleTranslator
import numpy as np
import ffmpeg

class SpeechToText:
    def __init__(self):
        self.model = whisper.load_model("base")
        self.langauges = {
            "US": "en",
            "UK": "en",
            "DE": "de",
            "DK": "da"}

    def inference(self, input, language_from, language_to):
        waveform = self.webm_to_waveform(input)
        transcription = self.model.transcribe(
            waveform, 
            language=self.langauges[language_from], 
            patience=2, 
            beam_size=5)["text"]
        if language_from == language_to:
            return transcription
        return GoogleTranslator(source=self.langauges[language_from], target=self.langauges[language_to]).translate(transcription)
    
    def webm_to_waveform(self, webm_bytes): 
        input = ffmpeg.input('pipe:',threads=0)
        audio = input.audio # extract audio from .webm video bytes
        out, _ = ffmpeg \
            .output(audio, 'pipe:', format='s16le', acodec='pcm_s16le', ar='16k', ac=1) \
            .run(capture_stdout = True, input=webm_bytes)
        waveform = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0 # div by 32768 => normalize by dividing by int16's max value
        return waveform
