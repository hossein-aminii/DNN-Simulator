from config import Config
from model.utils import ModelLoader
from .base_fault_injection import BaseFaultInjector
import os


class FaultInjector:

    def __init__(self, config: Config, name="A module to inject faults into the model") -> None:
        self.name = name
        self.config = config

        self.model_loader = ModelLoader(config=self.config)
        self.fault_injector_config = self.config.fault_injector_config
        self.int_bits = self.fault_injector_config["int_bits"]
        self.fraction_bits = self.fault_injector_config["fraction_bits"]
        self.total_bits = self.int_bits + self.fraction_bits
        self.fault_injection_ratio = self.fault_injector_config["fault_injection_ratio"]
        self.fix_point_format = self.fault_injector_config["fix_point_format"]
        self.mode = self.fault_injector_config["mode"]
        self.important_tensors = self.fault_injector_config["important_tensors"]
        self.results_directory = self.fault_injector_config["model_results_directory"]
        self.num_sample = self.fault_injector_config["num_sample"]
        if not os.path.exists(self.results_directory):
            os.mkdir(self.results_directory)

        print(
            f"INFO: start {self.total_bits}-bit fixed-point fault injection with int_bits: {self.int_bits} and fraction_bits: {self.fraction_bits}"
        )

    def run(self):
        fault_injector = BaseFaultInjector(
            fault_injection_ratio=self.fault_injection_ratio,
            int_bits=self.int_bits,
            fraction_bits=self.fraction_bits,
            fix_point_format=self.fix_point_format,
            mode=self.mode,
        )
        for sample_idx in range(self.num_sample):
            model = self.model_loader.load()
            print(f"INFO: fault injection in step {sample_idx + 1}/{self.num_sample}...")
            for layer in model.layers:
                if layer.name not in list(self.important_tensors.keys()):
                    continue
                # print(f"INFO: injecting faults in weights of layer {layer.name}...")
                weights = layer.get_weights()
                indexes = self.important_tensors[layer.name]
                new_layer_weights = []
                for idx, w in enumerate(weights):
                    if idx in indexes and len(w.shape) > 1:
                        faulty_w = fault_injector.inject_faults(weights=w)
                        new_layer_weights.append(faulty_w)
                    else:
                        new_layer_weights.append(w)
                layer.set_weights(new_layer_weights)
            model_path = os.path.join(self.results_directory, f"faulty_model#{sample_idx + 1}.h5")
            model.save(model_path)
            # print(f"INFO: model with fixed-point weights saved in {self.results_file_path}")
