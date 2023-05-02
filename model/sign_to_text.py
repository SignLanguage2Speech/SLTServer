# from model.model import ModelWrapper
from model import ModelWrapper
from Sign2Text.utils.load_model_from_checkpoint import load_model_from_checkpoint


class SignToText(ModelWrapper):
    def __init__(self):
        self.model = load_model_from_checkpoint('model/Sign2Text_test.pt', train=False)
        self.langauges = {
            "US": "en",
            "UK": "en",
            "DE": "de",
            "DK": "da"}
    def inference(self, rawvideo_input, language_from, language_to):
        return self.model.predict(rawvideo_input)


if __name__ == '__main__':
    s2t = SignToText()
    pass