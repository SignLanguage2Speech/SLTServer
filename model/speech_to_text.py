import whisper
import numpy as np
import ffmpeg

class SpeechToText:
    def __init__(self):
        self.model = whisper.load_model("base")
        self.langauges = {
            "english": "en",
            "german": "de",
            "danish": "da"}

    def inference(self, input, language):
        waveform = self.webm_to_waveform(input)
        return self.model.transcribe(
            waveform, 
            language=self.langauges[language], 
            patience=2, 
            beam_size=5)["text"]
    
    def webm_to_waveform(self, bytes): 
        input = ffmpeg.input('pipe:',threads=0)
        audio = input.audio.filter("aecho")
        out, _ = ffmpeg.output(audio, 'pipe:', format='s16le', acodec='pcm_s16le', ar='16k', ac=1).run(capture_stdout = True, input=bytes)
        waveform = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
        return waveform
