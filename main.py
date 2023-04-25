from model.model import TestModel
from server.server import ModelServer
from model.utils import Logger as LOG
from signals.speech_to_text import SpeechToText

if __name__ == '__main__':
    LOG.toggle_dbg(debug=True)
    # LOG.toggle_dbg(debug=False)
    slt_model = TestModel()
    tts_model = SpeechToText()
    model_server = ModelServer(slt_model,tts_model)
    model_server.run()