from config import Config
from model.utils import ModelLoader
from quantization.utils import FixedPointConverter


class FixPointPTQuantizer:

    def __init__(self, config: Config, name="A module to quantize a model in fix-point format") -> None:
        self.name = name
        self.config = config

        model_loader = ModelLoader(config=self.config)
        self.model = model_loader.load()
        self.quantizer_config = self.config.quantizer_config
        self.int_bits = self.quantizer_config["int_bits"]
        self.fraction_bits = self.quantizer_config["fraction_bits"]
        self.total_bits = self.quantizer_config["total_bits"]
        self.fix_point_format = self.quantizer_config["format"]
        self.important_tensors = self.quantizer_config["important_tensors"]
        self.results_file_path = self.quantizer_config["model_results_filepath"]

        print(
            f"INFO: start {self.total_bits}-bit fixed-point quantization with int_bits: {self.int_bits} and fraction_bits: {self.fraction_bits}"
        )

    def run(self):
        fix_point_converter = FixedPointConverter(int_bits=self.int_bits, frac_bits=self.fraction_bits)
        for layer in self.model.layers:
            if layer.name not in list(self.important_tensors.keys()):
                continue
            print(f"INFO: converting weights of layer {layer.name} into fixed-point format...")
            weights = layer.get_weights()
            indexes = self.important_tensors[layer.name]
            new_layer_weights = []
            for idx, w in enumerate(weights):
                if idx in indexes:
                    fixed_w = fix_point_converter.float_to_fix(array=w, format_type=self.fix_point_format)
                    new_layer_weights.append(fixed_w)
                else:
                    new_layer_weights.append(w)
            layer.set_weights(new_layer_weights)
            print(f"INFO: weights of layer {layer.name} converted to fixed-point format successfully!")

        self.model.save(self.results_file_path)
        print(f"INFO: model with fixed-point weights saved in {self.results_file_path}")
