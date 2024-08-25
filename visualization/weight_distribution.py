from .utils import DistributionVisualizer
from config import Config
from model.utils import ModelLoader
import numpy as np


class WeightDistributionVisualizer:

    def __init__(
        self,
        config: Config,
        name="A module to get the model as input and extract layer weights and for each, draw weight distribution",
    ) -> None:
        self.name = name
        self.config = config

        model_loader = ModelLoader(config=self.config)
        self.model = model_loader.load()
        self.visualizer_config = self.config.visualizer_config
        self.important_layers = self.visualizer_config.get("important_layers", [])
        print(f"INFO: important layers set to: {self.important_layers}")

    def extract_layer_name(self, tensor_name: str):
        try:
            return tensor_name.split("/")[0]
        except:
            return "Unknown"

    def get_tensor_type(self, tensor: np.ndarray):
        if len(tensor.shape) > 1:
            return "kernel"
        return "bias"

    def get_tensor_name(self, tensor_full_name: str):
        try:
            return tensor_full_name.split(":")[0].split("/")[-1]
        except:
            return "Unknown"

    def run(self):
        for w in self.model.weights:
            layer_name = self.extract_layer_name(tensor_name=w.name)
            tensor_type = self.get_tensor_type(tensor=w)
            if layer_name not in self.important_layers or tensor_type == "bias":
                continue
            tensor_name = self.get_tensor_name(tensor_full_name=w.name)
            print(f"Plotting weights disctibution of {layer_name}/{tensor_name}")
            weights_array: np.ndarray = w.numpy()
            visualizer = DistributionVisualizer(
                weights=weights_array, title=f"Weight Distribution for {layer_name}/{tensor_name}"
            )
            visualizer.plot_weight_distribution()
