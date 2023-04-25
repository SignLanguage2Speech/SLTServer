from model.model import TestModel
from server.server import ModelServer
from model.speech_to_text import SpeechToText

if __name__ == '__main__':
    model = TestModel()
    tts_model = SpeechToText()
    model_server = ModelServer(model,tts_model)
    model_server.run()