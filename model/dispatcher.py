from .lstm import LSTMModel


class ModelDispatcher:

    def __init__(self, model_type: str, name="A modudle to select the right model") -> None:
        self.name = name
        self.model_type = model_type

    def dispatch(self):
        if self.model_type == "LSTM":
            return LSTMModel()
        else:
            raise Exception(f"ERROR: Invalid or unsupported model type: {self.model_type}")
