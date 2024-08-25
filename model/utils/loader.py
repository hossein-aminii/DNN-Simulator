from config import Config
from model import ModelDispatcher


class ModelLoader:

    def __init__(self, config: Config, name="A module to load keras model") -> None:
        self.name = name
        self.config = config
        self.model_info = self.config.model_info

    def load(self):
        model_type = self.model_info["type"]
        dispatcher = ModelDispatcher(model_type=model_type)
        model_utils = dispatcher.dispatch()
        # --------------------------------------------------
        initialize = self.model_info["initialize"]
        if initialize:
            layers_info = self.model_info["layers_info"]
            return model_utils.crete_model(layers_info=layers_info)
        else:
            filepath = self.model_info["filepath"]
            return model_utils.load_model(filepath=filepath)
