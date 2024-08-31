from config import Config
from train import CustomTrainer
from inference import Inference
from visualization.visualizer_dispatcher import VisualizerDispatcher
from quantization.quantization_dispatcher import QuantizationDispatcher
from model.utils import ModelLoader
import numpy as np
from fault_injection.fault_injection_dispatcher import FaultInjectionDispatcher
from inference.manual_inference import ManualInference


class LSTMSimulator:

    def __init__(self, config: Config, name="A simulator for LSTM models") -> None:
        self.name = name
        self.config = config

    def train(self):
        trainer = CustomTrainer(config=self.config)
        trainer.train()

    def inference(self):
        inference_obj = Inference(config=self.config)
        inference_obj.inference()

    def quantization(self):
        quantizer = QuantizationDispatcher(config=self.config).dispatch()
        quantizer.run()

    def visualize(self):
        visualizer = VisualizerDispatcher(config=self.config).dispatch()
        visualizer.run()

    def test(self):
        model = ModelLoader(config=self.config).load()
        for layer in model.layers:
            if layer.name == "lstm":
                for w in layer.get_weights():
                    if len(w.shape) > 1:
                        for idx in np.ndindex(w.shape):
                            value = w[idx]
                            if str(value) != "0.0" and str(value) != "0.25" and str(value) != "-0.25":
                                print(f"value: {w[idx]}")

    def fault_injection(self):
        fault_injector = FaultInjectionDispatcher(config=self.config).dispatch()
        fault_injector.run()

    def manual_inference(self):
        inference_engine = ManualInference(config=self.config)
        inference_engine.run()

    def run(self):
        print("INFO: start running LSTM simulator...")
        if self.config.action == "train":
            self.train()
        elif self.config.action == "inference":
            self.inference()
        elif self.config.action == "quantization":
            self.quantization()
        elif self.config.action == "visualization":
            self.visualize()
        elif self.config.action == "test":
            self.test()
        elif self.config.action == "fault-injection":
            self.fault_injection()
        elif self.config.action == "manual-inference":
            self.manual_inference()
        else:
            raise Exception(
                f"ERROR: Invalid simulator action. The action {self.config.action} not supported or is invalid."
            )
        print("INFO: Running LSTM simulator done successfully!")
