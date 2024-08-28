import tensorflow as tf
from model import ModelDispatcher
from config import Config
from datasets import DataLoaderDispatcher
import pickle


class Inference:

    def __init__(self, config: Config, name="A module to train a model using custom training loop") -> None:
        self.name = name
        self.config = config
        self.model = self.get_model(info=self.config.model_info)
        self.data_loader = self.get_data_loader(info=self.config.dataset_info)

        self.train_dataset = self.data_loader.train_dataset
        self.validation_dataset = self.data_loader.validation_dataset
        self.inference_data = self.config.inference_data

        self.loss_function = self.get_loss_function()
        self.optimizer = self.get_optimizer()
        # weights = self.get_available_weights(indicator=4)
        # layers = {"lstm": [0, 1], "dense": [0]}
        # for layer in self.model.layers:
        #     if layer.name not in layers.keys():
        #         continue
        #     indexes = layers[layer.name]
        #     print(layer.name)
        #     for idx, w in enumerate(layer.get_weights()):
        #         if idx in indexes:
        #             flat_w = w.flatten()
        #             for value in flat_w:
        #                 if value not in weights:
        #                     print(f"ERROR: value {value} not in weights!")

    # def get_available_weights(self, indicator: int):
    #     weights = [0]
    #     num_values = 2 ** (indicator - 1)
    #     for idx in range(num_values - 1):
    #         weights.append(2 ** -(idx + 1))
    #         weights.append(-(2 ** -(idx + 1)))
    #     weights = np.array(weights)
    #     weights.sort()
    #     return weights

    def get_loss_function(self):
        return tf.keras.losses.CategoricalCrossentropy()

    def get_optimizer(self):
        return tf.keras.optimizers.Adam()

    def get_model(self, info: dict):
        model_type = info["type"]
        dispatcher = ModelDispatcher(model_type=model_type)
        model_utils = dispatcher.dispatch()
        # --------------------------------------------------
        initialize = info["initialize"]
        if initialize:
            layers_info = info["layers_info"]
            return model_utils.crete_model(layers_info=layers_info)
        else:
            filepath = info["filepath"]
            return model_utils.load_model(filepath=filepath)

    def load_tokenizer(self, path: str):
        try:
            with open(path, "rb") as handle:
                tokenizer = pickle.load(handle)
            print("INFO: Loading tokenizer successfully done!")
        except:
            tokenizer = None
            print("INFO: Can not load tokenizer. tokenizer set to None!")
        return tokenizer

    def get_data_loader(self, info: dict):
        dataset = info["name"]
        file_path = info["filepath"]
        tokenizer = self.load_tokenizer(path=info["tokenizer_filepath"])
        dispatcher = DataLoaderDispatcher(dataset=dataset)
        return dispatcher.dispatch(filepath=file_path, params={"tokenizer": tokenizer})

    def inference(self):
        # print(f"INFO: Start inference on {self.inference_data} data...")
        # Validation loop
        dataset = self.train_dataset if self.inference_data == "train" else self.validation_dataset
        progbar = tf.keras.utils.Progbar(len(dataset))
        loss_metric = tf.keras.metrics.Mean()
        accuracy_metric = tf.keras.metrics.CategoricalAccuracy()
        for step, (x_batch, y_batch) in enumerate(dataset):
            pred = self.model(x_batch, training=False)
            loss = self.loss_function(y_batch, pred)

            # Update validation metrics
            loss_metric.update_state(loss)
            accuracy_metric.update_state(y_batch, pred)

            progbar.update(
                step + 1,
                [
                    ("loss", loss_metric.result().numpy()),
                    ("accuracy", accuracy_metric.result().numpy()),
                ],
            )
        # print(f"INFO: inference on {self.inference_data} data done successfully!")
        return {"loss": float(loss_metric.result().numpy()), "accuracy": float(accuracy_metric.result().numpy())}
