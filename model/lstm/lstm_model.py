import tensorflow as tf
from tensorflow import keras


class LSTMModel:

    def __init__(self, name="A module to work with LSTM models") -> None:
        self.name = name

    @staticmethod
    def get_sequential_model():
        return keras.models.Sequential()

    @staticmethod
    def get_lstm_layer(info: dict):
        return keras.layers.LSTM(**info)

    @staticmethod
    def get_dense_layer(info: dict):
        return keras.layers.Dense(**info)

    def get_embedding_layer(self, info: dict):
        return keras.layers.Embedding(**info)

    def get_spatial_dropout_1d_layer(self, info: dict):
        return keras.layers.SpatialDropout1D(**info)

    def get_layer(self, info: dict):
        layer_type = info.pop("type")
        if layer_type == "LSTM":
            return self.get_lstm_layer(info=info)
        elif layer_type == "Dense":
            return self.get_dense_layer(info=info)
        elif layer_type == "Embedding":
            return self.get_embedding_layer(info=info)
        elif layer_type == "SpatialDropout1D":
            return self.get_spatial_dropout_1d_layer(info=info)
        else:
            raise Exception(f"ERROR: Invalid layer type. the layer type {layer_type} is not supported or is invalid.")

    def crete_model(self, layers_info: list):
        print("INFO: creating LSTM model...")
        model = self.get_sequential_model()
        for layer_info in layers_info:
            layer = self.get_layer(info=layer_info)
            model.add(layer)
        print("INFO: LSTM model created successfully. Here is the summary of the model:")
        print(model.summary())
        return model

    def load_model(self, filepath: str):
        return keras.models.load_model(filepath)
