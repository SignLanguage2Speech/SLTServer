from server.server import ModelServer
from model.utils import Logger as LOG
from model.speech_to_text import SpeechToText
from model.sign_to_text import SignToText

# from model.sign_to_text import main

if __name__ == '__main__':
    # main()
    LOG.debug = True
    LOG.measure_time = True
    slt_model = SignToText(device='cpu')
    tts_model = SpeechToText()
    model_server = ModelServer(slt_model,tts_model)
    model_server.run()

    # from media_processing.video import load_mp4video_from_file
    # import cv2, numpy as np
    # vid = load_mp4video_from_file('processed_output.mp4')
    # num_frames = vid.shape[0]
    # vid = model_server.pipeline(vid)
    # model_server.slt_model(vid, num_frames)