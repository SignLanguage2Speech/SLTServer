import ffmpeg, numpy as np

def webm_to_waveform(webm_bytes): 
        input = ffmpeg.input('pipe:',threads=0)
        audio = input.audio # extract audio from .webm video bytes
        out, _ = ffmpeg \
            .output(audio, 'pipe:', format='s16le', acodec='pcm_s16le', loglevel='quiet', ar='16k', ac=1) \
            .run(capture_stdout = True, input=webm_bytes)
        waveform = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0 # div by 32768 => normalize by dividing by int16's max value
        return waveform