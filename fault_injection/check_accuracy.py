from config import Config
from model.utils import ModelLoader
from inference import Inference
import os
import numpy as np
import json


class AccuracyChecker:

    def __init__(self, config: Config, name="A module to check and save accuracy of faulty models") -> None:
        self.name = name
        self.config = config

        self.model_loader = ModelLoader(config=self.config)
        self.fault_injector_config = self.config.fault_injector_config
        self.accuracy_results_filename = self.fault_injector_config["accuracy_results_filename"]
        self.results_directory = self.fault_injector_config["model_results_directory"]
        if not os.path.exists(self.results_directory):
            os.mkdir(self.results_directory)

    def get_model_paths(self, directory: str):
        return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".h5")]

    def save_accuracy_results(self, results: dict):
        path = os.path.join(self.results_directory, self.accuracy_results_filename)
        with open(path, "w") as jf:
            json.dump(results, jf)

    def run(self):
        model_paths = self.get_model_paths(directory=self.results_directory)
        inference = Inference(config=self.config)
        results = self.fault_injector_config
        accuracy_details = {"loss": [], "accuracy": [], "model_path": []}
        for idx, model_path in enumerate(model_paths):
            print(f"check accuracy step {idx + 1}/{len(model_paths)}")
            new_model_info = {"type": "LSTM", "initialize": False, "filepath": model_path}
            self.model_loader.model_info = new_model_info
            model = self.model_loader.load()
            inference.model = model
            inference_results = inference.inference()
            accuracy_details["loss"].append(inference_results["loss"])
            accuracy_details["accuracy"].append(inference_results["accuracy"])
            accuracy_details["model_path"].append(model_path)
            results["accuracy_details"] = accuracy_details
            results["mean_loss"] = float(np.mean(accuracy_details["loss"]))
            results["mean_accuracy"] = float(np.mean(accuracy_details["accuracy"]))
            self.save_accuracy_results(results=results)
            model = None
