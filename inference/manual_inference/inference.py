from config import Config
from model.utils import ModelLoader
from datasets import DataLoaderDispatcher
import pickle
import random
import numpy as np
import tensorflow as tf


class ManualInference:

    def __init__(self, config: Config, name="A module to do inference of model manually!"):
        self.name = name

        self.config = config
        model_loader = ModelLoader(config=self.config)
        self.model = model_loader.load()

        # self.data_loader = self.get_data_loader(info=self.config.dataset_info)
        # self.train_dataset = self.data_loader.train_dataset
        # self.validation_dataset = self.data_loader.validation_dataset

        self.manual_inference_config = self.config.manual_inference_config
        self.target_layer = self.manual_inference_config["target_layer"]
        self.prev_target_layer = self.manual_inference_config["prev_target_layer"]
        self.weights_info = self.manual_inference_config["weights_info"]
        self.bias_info = self.manual_inference_config["bias_info"]

        self.target_layer_parameters = {}

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

    # def get_random_input(self, _from="train"):
    #     if _from == "train":
    #         dataset = self.train_dataset
    #     else:
    #         dataset = self.validation_dataset
    #     dataset_list = list(dataset.unbatch().as_numpy_iterator())
    #     input_idx = random.randint(0, len(dataset_list) - 1)
    #     return dataset_list[input_idx][0]

    def get_random_input(self):
        shape = (128,)
        return np.random.random(shape)

    def extract_target_layer_parameters(self, target_layer):
        for idx, weight in enumerate(target_layer.get_weights()):
            if idx in list(self.bias_info.keys()):
                key = self.bias_info[idx]
                self.target_layer_parameters[key] = weight
            elif idx in list(self.weights_info.keys()):
                key = self.weights_info[idx]
                self.target_layer_parameters[key] = weight

    def get_tf_results(self, model_input: np.ndarray):
        model_input = model_input.reshape((1, model_input.shape[0]))  # for spatial dropout!
        input_layer = self.model.input
        target_layer_input = self.model.get_layer(name=self.prev_target_layer).output
        target_layer = self.model.get_layer(name=self.target_layer)
        self.extract_target_layer_parameters(target_layer=target_layer)
        temp_model = tf.keras.Model(inputs=input_layer, outputs=[target_layer_input, target_layer.output])
        target_layer_input, target_layer_output = temp_model.predict(model_input)

        return self.adjust_shape(target_layer_input), self.adjust_shape(target_layer_output)

    def adjust_shape(self, array: np.ndarray):
        target_shape = []
        for dim in array.shape:
            if dim > 1:
                target_shape.append(dim)
        return array.reshape(target_shape)

    def get_manual_results(self, layer_input: np.ndarray):
        biases: np.ndarray = self.target_layer_parameters["biases"]
        kernel: np.ndarray = self.target_layer_parameters["kernel"]
        recurrent_kernel: np.ndarray = self.target_layer_parameters["recurrent_kernel"]
        print(
            f"INFO: input shape: {layer_input.shape}, bias shape: {biases.shape}, kernel shape: {kernel.shape}, recurrent kernel shape: {recurrent_kernel.shape}"
        )

    def run(self):
        print("INFO: manual inference called!")
        # model_input: np.ndarray = self.get_random_input(_from="validation")
        # print(model_input.shape)
        model_input: np.ndarray = self.get_random_input()
        tf_target_input, tf_target_output = self.get_tf_results(model_input=model_input)
        print(f"INFO: TF Target input shape: {tf_target_input.shape}, TF Target output shape: {tf_target_output.shape}")
        manual_target_output = self.get_manual_results(layer_input=tf_target_input)
