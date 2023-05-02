from server.server import ModelServer
from model.utils import Logger as LOG
from model.speech_to_text import SpeechToText
from model.sign_to_text import SignToText

# from model.sign_to_text import main

if __name__ == '__main__':
    # main()
    LOG.debug = True
    LOG.measure_time = True
    slt_model = SignToText()
    tts_model = SpeechToText()
    model_server = ModelServer(slt_model,tts_model)
    model_server.run()